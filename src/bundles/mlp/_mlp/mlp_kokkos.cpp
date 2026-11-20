/*
 * mlp_kokkos.cpp
 * Combined Kokkos implementation for ChimeraX MLP
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>

// -----------------------------------------------------------------------------
// Constants and Typedefs
// -----------------------------------------------------------------------------

enum MLPMethod {
    METHOD_FAUCHERE = 0,
    METHOD_BRASSEUR = 1,
    METHOD_BUCKINGHAM = 2,
    METHOD_DUBOST = 3,
    METHOD_TYPE5 = 4
};

// Global state tracking
static bool is_kokkos_initialized = false;

// -----------------------------------------------------------------------------
// Kernel Functor
// -----------------------------------------------------------------------------

template<class DeviceType>
struct MLPKernel {
    using ExecutionSpace = typename DeviceType::execution_space;
    using MemorySpace = typename DeviceType::memory_space;

    // Device Views (accessed via DualView on device side)
    // We use LayoutLeft (Column-Major) on device for potential coalescing benefits
    Kokkos::View<const float*[3], Kokkos::LayoutLeft, MemorySpace> d_xyz;
    Kokkos::View<const float*, Kokkos::LayoutLeft, MemorySpace> d_fi;
    Kokkos::View<float***, Kokkos::LayoutLeft, MemorySpace> d_pot;

    float x0, y0, z0;
    float spacing;
    float max_dist;
    int nz, ny, nx;
    int method;
    float nexp;
    int md_steps;

    MLPKernel(
        Kokkos::View<const float*[3], Kokkos::LayoutLeft, MemorySpace> xyz,
        Kokkos::View<const float*, Kokkos::LayoutLeft, MemorySpace> fi,
        Kokkos::View<float***, Kokkos::LayoutLeft, MemorySpace> pot,
        const float* origin, float _spacing, float _max_dist, 
        int _method, float _nexp
    ) : d_xyz(xyz), d_fi(fi), d_pot(pot),
        spacing(_spacing), max_dist(_max_dist), method(_method), nexp(_nexp)
    {
        x0 = origin[0];
        y0 = origin[1];
        z0 = origin[2];
        nz = pot.extent(0);
        ny = pot.extent(1);
        nx = pot.extent(2);
        md_steps = (int)std::ceil(max_dist / spacing);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& a) const {
        float ax = d_xyz(a, 0);
        float ay = d_xyz(a, 1);
        float az = d_xyz(a, 2);
        float f = 100.0f * d_fi(a);

        // Calculate grid bounds for this atom
        float i0_f = (ax - x0) / spacing;
        float j0_f = (ay - y0) / spacing;
        float k0_f = (az - z0) / spacing;

        int kmin = (int)Kokkos::floor(k0_f - md_steps); if(kmin < 0) kmin = 0;
        int kmax = (int)Kokkos::ceil(k0_f + md_steps);  if(kmax >= nz) kmax = nz - 1;

        int jmin = (int)Kokkos::floor(j0_f - md_steps); if(jmin < 0) jmin = 0;
        int jmax = (int)Kokkos::ceil(j0_f + md_steps);  if(jmax >= ny) jmax = ny - 1;

        int imin = (int)Kokkos::floor(i0_f - md_steps); if(imin < 0) imin = 0;
        int imax = (int)Kokkos::ceil(i0_f + md_steps);  if(imax >= nx) imax = nx - 1;

        for (int k = kmin; k <= kmax; ++k) {
            float gz = z0 + k * spacing;
            float dz = az - gz;
            float dz2 = dz * dz;

            for (int j = jmin; j <= jmax; ++j) {
                float gy = y0 + j * spacing;
                float dy = ay - gy;
                float dy2 = dy * dy;

                for (int i = imin; i <= imax; ++i) {
                    float gx = x0 + i * spacing;
                    float dx = ax - gx;
                    // d = distance
                    float d = Kokkos::sqrt(dx*dx + dy2 + dz2);

                    if (d <= max_dist) {
                        float p = 0.0f;
                        if (method == METHOD_FAUCHERE) p = Kokkos::exp(-d);
                        else if (method == METHOD_BRASSEUR) p = Kokkos::exp(-d / 3.1f);
                        else if (method == METHOD_BUCKINGHAM) p = (d > 1.0e-6f) ? (1.0f / Kokkos::pow(d, nexp)) : 0.0f;
                        else if (method == METHOD_DUBOST) p = 1.0f / (1.0f + d);
                        else if (method == METHOD_TYPE5) p = Kokkos::exp(-Kokkos::sqrt(d));

                        Kokkos::atomic_add(&d_pot(k, j, i), f * p);
                    }
                }
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Host Dispatcher (Internal)
// -----------------------------------------------------------------------------

template <class DeviceType>
void run_mlp_device(
    int n_atoms, const float* h_xyz_ptr, const float* h_fi_ptr,
    int nz, int ny, int nx, float* h_pot_ptr,
    const float* origin, float spacing, float max_dist,
    int method, float nexp) 
{
    // 1. Create DualViews
    // LayoutLeft on Device (GPU friendly), LayoutRight on Host (NumPy friendly)
    Kokkos::DualView<float*[3], Kokkos::LayoutLeft, DeviceType> dv_atoms("atoms", n_atoms);
    Kokkos::DualView<float*, Kokkos::LayoutLeft, DeviceType> dv_fi("fi", n_atoms);
    Kokkos::DualView<float***, Kokkos::LayoutLeft, DeviceType> dv_pot("pot", nz, ny, nx);

    // 2. Wrap Host Pointers (Unmanaged View around NumPy data)
    Kokkos::View<const float*[3], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        h_atoms_wrap(h_xyz_ptr, n_atoms);
    Kokkos::View<const float*, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        h_fi_wrap(h_fi_ptr, n_atoms);
    // Note: We use the existing pot array to avoid allocation if possible, or copy back later
    Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        h_pot_wrap(h_pot_ptr, nz, ny, nx);

    // 3. Deep Copy Host -> Device
    dv_atoms.modify_host();
    dv_atoms.sync_device(); // If on CPU, this might be no-op; on GPU it triggers copy
    // However, since dv_atoms was just created, we need to copy FROM the wrapper:
    Kokkos::deep_copy(dv_atoms.view_device(), h_atoms_wrap);
    
    dv_fi.modify_host();
    Kokkos::deep_copy(dv_fi.view_device(), h_fi_wrap);

    // Initialize Pot to 0 on device
    Kokkos::deep_copy(dv_pot.view_device(), 0.0f);
    dv_pot.modify_device(); // Data is now valid on device

    // 4. Execute Kernel
    MLPKernel<DeviceType> kernel(
        dv_atoms.view_device(),
        dv_fi.view_device(),
        dv_pot.view_device(),
        origin, spacing, max_dist, method, nexp
    );

    Kokkos::parallel_for(
        "MLP_Kernel",
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, n_atoms),
        kernel
    );
    
    Kokkos::fence();

    // 5. Deep Copy Device -> Host
    // Copy result back to the Python-provided pointer
    Kokkos::deep_copy(h_pot_wrap, dv_pot.view_device());
}

// -----------------------------------------------------------------------------
// C Exported Functions (for Cython)
// -----------------------------------------------------------------------------

extern "C" {

void ensure_kokkos_initialized() {
    if (!is_kokkos_initialized && !Kokkos::is_initialized()) {
        Kokkos::initialize();
        is_kokkos_initialized = true;
    }
}

void finalize_kokkos() {
    if (Kokkos::is_initialized()) {
        Kokkos::finalize();
        is_kokkos_initialized = false;
    }
}

void run_mlp_kokkos_main(
    int n_atoms, const float* h_xyz_ptr, const float* h_fi_ptr,
    int nz, int ny, int nx, float* h_pot_ptr,
    const float* origin, float spacing, float max_dist,
    int method, float nexp)
{
    ensure_kokkos_initialized();

    // Dispatch to the default execution space configured at compile time.
    // If compiled with OpenMP, this runs on CPU threads.
    // If compiled with CUDA/HIP, this runs on GPU.
    run_mlp_device<Kokkos::DefaultExecutionSpace>(
        n_atoms, h_xyz_ptr, h_fi_ptr,
        nz, ny, nx, h_pot_ptr,
        origin, spacing, max_dist,
        method, nexp
    );
}

} // extern "C"
