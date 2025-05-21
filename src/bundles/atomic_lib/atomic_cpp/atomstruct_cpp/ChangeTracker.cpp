// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "ChangeTracker.h"
#include <pyinstance/PythonInstance.instantiate.h>

template class pyinstance::PythonInstance<atomstruct::ChangeTracker>;

namespace atomstruct {

const std::string ChangeTracker::REASON_ACTIVE_COORD_SET("active_coordset changed");
const std::string ChangeTracker::REASON_ALT_LOC("alt_loc changed");
const std::string ChangeTracker::REASON_ALT_LOCS("alt_locs changed");
const std::string ChangeTracker::REASON_ANISO_U("aniso_u changed");
const std::string ChangeTracker::REASON_BALL_SCALE("ball_scale changed");
const std::string ChangeTracker::REASON_BFACTOR("bfactor changed");
const std::string ChangeTracker::REASON_CHAIN_ID("chain_id changed");
const std::string ChangeTracker::REASON_COLOR("color changed");
const std::string ChangeTracker::REASON_COORD("coord changed");
const std::string ChangeTracker::REASON_COORDSET("coordset changed");
const std::string ChangeTracker::REASON_DISPLAY("display changed");
const std::string ChangeTracker::REASON_DRAW_MODE("draw_mode changed");
const std::string ChangeTracker::REASON_ELEMENT("element changed");
const std::string ChangeTracker::REASON_HALFBOND("halfbond changed");
const std::string ChangeTracker::REASON_HIDE("hide changed");
const std::string ChangeTracker::REASON_IDATM_TYPE("idatm_type changed");
const std::string ChangeTracker::REASON_INSERTION_CODE("insertion_code changed");
const std::string ChangeTracker::REASON_NAME("name changed");
const std::string ChangeTracker::REASON_NUMBER("number changed");
const std::string ChangeTracker::REASON_OCCUPANCY("occupancy changed");
const std::string ChangeTracker::REASON_RADIUS("radius changed");
const std::string ChangeTracker::REASON_RESIDUES("residues changed");
const std::string ChangeTracker::REASON_RIBBON_ADJUST("ribbon_adjust changed");
const std::string ChangeTracker::REASON_RIBBON_COLOR("ribbon_color changed");
const std::string ChangeTracker::REASON_RIBBON_DISPLAY("ribbon_display changed");
const std::string ChangeTracker::REASON_RIBBON_HIDE_BACKBONE("ribbon_hide_backbone changed");
const std::string ChangeTracker::REASON_RIBBON_TETHER("ribbon_tether_* changed");
const std::string ChangeTracker::REASON_RIBBON_ORIENTATION("ribbon_orientation changed");
const std::string ChangeTracker::REASON_RIBBON_MODE("ribbon_mode changed");
const std::string ChangeTracker::REASON_RING_COLOR("ring_color changed");
const std::string ChangeTracker::REASON_RING_DISPLAY("ring_display changed");
const std::string ChangeTracker::REASON_RING_MODE("ring_mode changed");
const std::string ChangeTracker::REASON_SCENE_COORD("scene_coord changed");
const std::string ChangeTracker::REASON_SELECTED("selected changed");
const std::string ChangeTracker::REASON_SEQUENCE("sequence changed");
const std::string ChangeTracker::REASON_SERIAL_NUMBER("serial_number changed");
const std::string ChangeTracker::REASON_STRUCTURE_CATEGORY("structure_category changed");
const std::string ChangeTracker::REASON_SS_ID("ss_id changed");
const std::string ChangeTracker::REASON_SS_TYPE("ss_type changed");
const std::string ChangeTracker::REASON_WORM_RADIUS("worm_radius changed");
const std::string ChangeTracker::REASON_WORM_RIBBON("worm_ribbon changed");

DiscardingChangeTracker  dct;

DiscardingChangeTracker*
DiscardingChangeTracker::discarding_change_tracker() { return &dct; }

}  // namespace atomstruct
