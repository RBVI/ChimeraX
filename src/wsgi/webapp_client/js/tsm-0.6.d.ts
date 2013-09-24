module TSM {
    class vec3 {
        private values;
        public x : number;
        public y : number;
        public z : number;
        public xy : number[];
        public xyz : number[];
        constructor (values?: number[]);
        public at(index: number): number;
        public reset(): void;
        public copy(dest?: vec3): vec3;
        public negate(dest?: vec3): vec3;
        public equals(vector: vec3, threshold?: number): bool;
        public length(): number;
        public squaredLength(): number;
        public add(vector: vec3): vec3;
        public subtract(vector: vec3): vec3;
        public multiply(vector: vec3): vec3;
        public divide(vector: vec3): vec3;
        public scale(value: number, dest?: vec3): vec3;
        public normalize(dest?: vec3): vec3;
        public multiplyMat3(matrix: mat3, dest?: vec3): vec3;
        public multiplyQuat(quat1: quat, dest?: vec3): vec3;
        static cross(vector: vec3, vector2: vec3, dest?: vec3): vec3;
        static dot(vector: vec3, vector2: vec3): number;
        static distance(vector: vec3, vector2: vec3): number;
        static squaredDistance(vector: vec3, vector2: vec3): number;
        static direction(vector: vec3, vector2: vec3, dest?: vec3): vec3;
        static interpolate(vector: vec3, vector2: vec3, time: number, dest?: vec3): vec3;
        static sum(vector: vec3, vector2: vec3, dest?: vec3): vec3;
        static difference(vector: vec3, vector2: vec3, dest?: vec3): vec3;
        static product(vector: vec3, vector2: vec3, dest?: vec3): vec3;
        static quotient(vector: vec3, vector2: vec3, dest?: vec3): vec3;
        public toQuat(dest?: quat): quat;
        static zero: vec3;
        static up: vec3;
        static right: vec3;
        static forward: vec3;
    }
}
module TSM {
    class vec4 {
        private values;
        public x : number;
        public y : number;
        public z : number;
        public w : number;
        public xy : number[];
        public xyz : number[];
        public xyzw : number[];
        public r : number;
        public g : number;
        public b : number;
        public a : number;
        public rg : number[];
        public rgb : number[];
        public rgba : number[];
        constructor (values?: number[]);
        public at(index: number): number;
        public reset(): void;
        public copy(dest?: vec4): vec4;
        public negate(dest?: vec4): vec4;
        public equals(vector: vec4, threshold?: number): bool;
        public length(): number;
        public squaredLength(): number;
        public add(vector: vec4): vec4;
        public subtract(vector: vec4): vec4;
        public multiply(vector: vec4): vec4;
        public divide(vector: vec4): vec4;
        public scale(value: number, dest?: vec4): vec4;
        public normalize(dest?: vec4): vec4;
        public multiplyMat4(matrix: mat4, dest?: vec4): vec4;
        static interpolate(vector: vec4, vector2: vec4, time: number, dest?: vec4): vec4;
        static sum(vector: vec4, vector2: vec4, dest?: vec4): vec4;
        static difference(vector: vec4, vector2: vec4, dest?: vec4): vec4;
        static product(vector: vec4, vector2: vec4, dest?: vec4): vec4;
        static quotient(vector: vec4, vector2: vec4, dest?: vec4): vec4;
        static zero: vec4;
    }
}
module TSM {
    class mat2 {
        private values;
        constructor (values?: number[]);
        public at(index: number): number;
        public init(values: number[]): mat2;
        public reset(): void;
        public copy(dest?: mat2): mat2;
        public all(): number[];
        public row(index: number): number[];
        public col(index: number): number[];
        public equals(matrix: mat2, threshold?: number): bool;
        public determinant(): number;
        public setIdentity(): mat2;
        public transpose(dest?: mat2): mat2;
        public inverse(dest?: mat2): mat2;
        public multiply(matrix: mat2, dest?: mat2): mat2;
        public rotate(angle: number, dest?: mat2): mat2;
        public multiplyVec2(vector: vec2, dest?: vec2): vec2;
        public scale(vector: vec2, dest?: mat2): mat2;
        static product(m1: mat2, m2: mat2, dest?: mat2): mat2;
        static identity: mat2;
    }
}
module TSM {
    class mat3 {
        private values;
        constructor (values?: number[]);
        public at(index: number): number;
        public init(values: number[]): mat3;
        public reset(): void;
        public copy(dest?: mat3): mat3;
        public all(): number[];
        public row(index: number): number[];
        public col(index: number): number[];
        public equals(matrix: mat3, threshold?: number): bool;
        public determinant(): number;
        public setIdentity(): mat3;
        public transpose(dest?: mat3): mat3;
        public inverse(dest?: mat3): mat3;
        public multiply(matrix: mat3, dest?: mat3): mat3;
        public multiplyVec2(vector: vec2, dest?: vec2): vec2;
        public multiplyVec3(vector: vec3, dest?: vec3): vec3;
        public toMat4(dest?: mat4): mat4;
        public toQuat(dest?: quat): quat;
        public rotate(angle: number, axis: vec3, dest?: mat3): mat3;
        static product(m1: mat3, m2: mat3, dest?: mat3): mat3;
        static identity: mat3;
    }
}
module TSM {
    class mat4 {
        private values;
        constructor (values?: number[]);
        public at(index: number): number;
        public init(values: number[]): mat4;
        public reset(): void;
        public copy(dest?: mat4): mat4;
        public all(): number[];
        public row(index: number): number[];
        public col(index: number): number[];
        public equals(matrix: mat4, threshold?: number): bool;
        public determinant(): number;
        public setIdentity(): mat4;
        public transpose(dest?: mat4): mat4;
        public inverse(dest?: mat4): mat4;
        public multiply(matrix: mat4, dest?: mat4): mat4;
        public multiplyVec3(vector: vec3, dest?: vec3): vec3;
        public multiplyVec4(vector: vec4, dest?: vec4): vec4;
        public toMat3(dest?: mat3): mat3;
        public toInverseMat3(dest?: mat3): mat3;
        public translate(vector: vec3, dest?: mat4): mat4;
        public scale(vector: vec3, dest?: mat4): mat4;
        public rotate(angle: number, axis: vec3, dest?: mat4): mat4;
        static frustum(left: number, right: number, bottom: number, top: number, near: number, far: number, dest?: mat4): mat4;
        static perspective(fov: number, aspect: number, near: number, far: number, dest?: mat4): mat4;
        static orthographic(left: number, right: number, bottom: number, top: number, near: number, far: number, dest?: mat4): mat4;
        static lookAt(position: vec3, target: vec3, up?: vec3, dest?: mat4): mat4;
        static product(m1: mat4, m2: mat4, dest?: mat4): mat4;
        static identity: mat4;
    }
}
module TSM {
    class quat {
        private values;
        public x : number;
        public y : number;
        public z : number;
        public w : number;
        public xy : number[];
        public xyz : number[];
        public xyzw : number[];
        constructor (values?: number[]);
        public at(index: number): number;
        public reset(): void;
        public copy(dest?: quat): quat;
        public roll(): number;
        public pitch(): number;
        public yaw(): number;
        public equals(vector: quat, threshold?: number): bool;
        public setIdentity(): quat;
        public calculateW(): quat;
        static dot(q1: quat, q2: quat): number;
        public inverse(dest?: quat): quat;
        public conjugate(dest?: quat): quat;
        public length(): number;
        public normalize(dest?: quat): quat;
        public add(other: quat): quat;
        public multiply(other: quat): quat;
        public multiplyVec3(vector: vec3, dest?: vec3): vec3;
        public toMat3(dest?: mat3): mat3;
        public toMat4(dest?: mat4): mat4;
        static sum(q1: quat, q2: quat, dest?: quat): quat;
        static product(q1: quat, q2: quat, dest?: quat): quat;
        static interpolate(q1: quat, q2: quat, time: number, dest?: quat): quat;
        static identity: quat;
    }
}
var EPSILON: number;
module TSM {
    class vec2 {
        private values;
        public x : number;
        public y : number;
        public xy : number[];
        constructor (values?: number[]);
        public at(index: number): number;
        public reset(): void;
        public copy(dest?: vec2): vec2;
        public negate(dest?: vec2): vec2;
        public equals(vector: vec2, threshold?: number): bool;
        public length(): number;
        public squaredLength(): number;
        public add(vector: vec2): vec2;
        public subtract(vector: vec2): vec2;
        public multiply(vector: vec2): vec2;
        public divide(vector: vec2): vec2;
        public scale(value: number, dest?: vec2): vec2;
        public normalize(dest?: vec2): vec2;
        public multiplyMat2(matrix: mat2, dest?: vec2): vec2;
        public multiplyMat3(matrix: mat3, dest?: vec2): vec2;
        static cross(vector: vec2, vector2: vec2, dest?: vec3): vec3;
        static dot(vector: vec2, vector2: vec2): number;
        static distance(vector: vec2, vector2: vec2): number;
        static squaredDistance(vector: vec2, vector2: vec2): number;
        static direction(vector: vec2, vector2: vec2, dest?: vec2): vec2;
        static interpolate(vector: vec2, vector2: vec2, time: number, dest?: vec2): vec2;
        static sum(vector: vec2, vector2: vec2, dest?: vec2): vec2;
        static difference(vector: vec2, vector2: vec2, dest?: vec2): vec2;
        static product(vector: vec2, vector2: vec2, dest?: vec2): vec2;
        static quotient(vector: vec2, vector2: vec2, dest?: vec2): vec2;
        static zero: vec2;
    }
}
