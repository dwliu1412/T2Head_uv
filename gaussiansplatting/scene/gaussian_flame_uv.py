#

#
import os
import trimesh
import numpy as np
from smplx import FLAME
from plyfile import PlyData, PlyElement

import torch
from torch import nn
import torch.nn.functional as F

from simple_knn._C import distCUDA2

from gaussiansplatting.utils.sh_utils import RGB2SH
from gaussiansplatting.utils.system_utils import mkdir_p
from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.utils.general_utils import strip_symmetric, build_scaling_rotation_only
from gaussiansplatting.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from gaussiansplatting.utils.net import UVMiniUNetSmoother, UVResSmoother, UVAtlasHeads

def _to_cpu_tensor(x):
    if x is None:
        return None
    if isinstance(x, nn.Parameter):
        x = x.data
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x

class GaussianFlameUVModel(GaussianModel):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            verts, vnormals = self._flame_verts_and_normals()

            J = self._gaussian_jacobian_J(verts, vnormals)
            Tu = J[:, :, 0]
            Tv = J[:, :, 1]
            eps = 1e-6
            tu_norm = Tu.norm(dim=-1).clamp_min(eps)
            tv_norm = Tv.norm(dim=-1).clamp_min(eps)

            # 缓存 detach 版本，避免无谓二阶/graph 膨胀（ratio_w 多数当权重用）
            self._cache_tu_norm = tu_norm.detach()
            self._cache_tv_norm = tv_norm.detach()
            self._cache_ratio_w = (tu_norm / tv_norm).detach()

            s = scaling_modifier * scaling  # (N,3)
            Rm = build_rotation(rotation)  # (N,3,3)
            Sigma_uvd = Rm @ torch.diag_embed(s * s) @ Rm.transpose(1, 2)  # (N,3,3)

            Sigma_xyz = J @ Sigma_uvd @ J.transpose(1, 2)
            sigma_floor = 1e-4  #
            Sigma_xyz = Sigma_xyz + torch.eye(3, device=Sigma_xyz.device)[None] * (sigma_floor ** 2)

            return strip_symmetric(Sigma_xyz)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, model_folder, gender='generic', device='cuda',
                 jacobian_mode: str = 'autodiff', jacobian_create_graph: bool = False,
                 update_faces_on_densify: bool = True):
        super().__init__(sh_degree)
        self.device = torch.device(device)
        self.jacobian_mode = jacobian_mode
        self.jacobian_create_graph = jacobian_create_graph
        # When densifying (split/clone), immediately re-assign face index from UV so points can move across triangles.
        self.update_faces_on_densify = update_faces_on_densify

        self.num_betas = 300
        self.num_expression = 100
        self.model = FLAME(
            model_folder,
            gender=gender,
            ext='pkl',
            num_betas=self.num_betas,
            num_expression_coeffs=self.num_expression,
            create_global_orient=True,
        ).to(self.device)

        self.flame_scale = 0
        self.densify_scale = 1
        self.center = 0
        self.scale = 0
        self.T = torch.empty(0)
        self.R = torch.empty(0)
        self.S = torch.empty(0)
        self._shape = torch.empty(0)
        self._expression = torch.empty(0)
        self._jaw_pose = torch.zeros([1, 3], device=self.device)
        self._leye_pose = torch.zeros([1, 3], device=self.device)
        self._reye_pose = torch.zeros([1, 3], device=self.device)
        self._neck_pose = torch.zeros([1, 3], device=self.device)

        # load FLAME texture
        tex = np.load(os.path.join(model_folder, 'FLAME_texture.npz'))
        vt = tex['vt']   # [5118, 2]，UV 坐标
        ft = tex['ft']   # [9976, 3]，每个三角面在 vt 中的索引
        # mean_tex = tex['mean'] # 平均纹理
        self.vt = torch.from_numpy(vt.astype(np.float32)).to(self.device)  # (T,2)
        self.ft = torch.from_numpy(ft.astype(np.int64)).to(self.device)  # (F,3)

        self._build_uv_grid(res=256, maxK=32)

        self.use_uv_atlas = True
        self.uv_atlas_res = 256
        self.uv_feat_dim = 64
        self.uv_feat = nn.Parameter(torch.zeros(1, self.uv_feat_dim, self.uv_atlas_res, self.uv_atlas_res, device=self.device))
        self.uv_smoother_type = "res"  #
        if self.uv_smoother_type == "unet":
            self.uv_smoother = UVMiniUNetSmoother(self.uv_feat_dim).to(self.device)
        else:
            self.uv_smoother = UVResSmoother(self.uv_feat_dim).to(self.device)

        self.uv_heads = UVAtlasHeads(self.uv_feat_dim, opacity_init=0.1).to(self.device)

    @property
    def get_shape(self):
        return self._shape

    @property
    def get_expression(self):
        return self._expression

    @property
    def get_faces(self):
        return self._faces

    @property
    def get_jaw_pose(self):
        return self._jaw_pose

    @property
    def get_leye_pose(self):
        return self._leye_pose

    @property
    def get_reye_pose(self):
        return self._reye_pose

    @property
    def get_neck_pose(self):
        return self._neck_pose

    @property
    def get_scaling(self):
        scaling = self.scaling_activation(self._scaling)
        scaling = torch.clamp(scaling, min=1e-5)
        # scaling = self.scaling_activation((S + 1e-10).sqrt().unsqueeze(-1) * self._scaling)
        return scaling

    def _uv_atlas_map(self):
        """返回经过 CNN/U-Net 内置平滑后的 UV feature map"""
        # 你也可以加一个 scale 控制 smoothing 强度：return self.uv_feat + 0.5*(self.uv_smoother(self.uv_feat)-self.uv_feat)
        return self.uv_smoother(self.uv_feat)

    def tbn(self, tris):
        # triangles: Tensor[num, 3, 3]
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        n = F.normalize(torch.cross(b - a, c - a), dim=-1)
        d = b - a

        X = F.normalize(torch.cross(d, n), dim=-1)
        Y = F.normalize(torch.cross(d, X), dim=-1)
        Z = F.normalize(d, dim=-1)

        return torch.stack([X, Y, Z], dim=1)

    @property
    def get_xyz(self):
        """Map learnable (u,v,d) to world-space xyz.
        Implements TeGA Eq.(1): p = p_surf(u,v) + d * n(u,v),
        where p_surf and n are barycentrically interpolated on the UV triangle.
        """
        verts, vnormals = self._flame_verts_and_normals()
        uvd = torch.cat([self.get_uv, self._d], dim=1)  # (N,3) with columns [u,v,d]
        return self._map_uvd_to_xyz(uvd, verts, vnormals)

    def _map_uvd_to_xyz(self, uvd, verts, vnormals, face_idx=None):
        """Core mapping F: (u,v,d) -> xyz for a fixed mesh (verts, vnormals).

        Args:
            uvd: (N,3) tensor in texture-space coordinates.
            verts: (V,3) FLAME vertices in world space.
            vnormals: (V,3) vertex normals in world space.
            face_idx: optional (N,) long tensor. If None uses self._face_idx.

        Returns:
            xyz: (N,3) world-space positions.
        """
        uv = uvd[:, :2]
        d = uvd[:, 2:3]
        if face_idx is None:
            face_idx = self._face_idx.long()
        faces = self.get_faces.long()  # (F,3) geometry
        # Ensure face indices are valid for both geometry and UV topology
        num_faces = min(faces.shape[0], self.ft.shape[0])
        f = face_idx.clamp(0, num_faces - 1)

        gtri = faces[f]         # (N,3) geometry vertex ids
        ttri = self.ft[f]       # (N,3) texture vertex ids

        p0 = verts[gtri[:, 0]]
        p1 = verts[gtri[:, 1]]
        p2 = verts[gtri[:, 2]]

        n0 = vnormals[gtri[:, 0]]
        n1 = vnormals[gtri[:, 1]]
        n2 = vnormals[gtri[:, 2]]

        uv0 = self.vt[ttri[:, 0]]
        uv1 = self.vt[ttri[:, 1]]
        uv2 = self.vt[ttri[:, 2]]

        w0, w1, w2 = self.barycentric_2d(uv, uv0, uv1, uv2)
        w0, w1, w2 = self.clamp_small_negatives(w0, w1, w2)

        psurf = w0[:, None] * p0 + w1[:, None] * p1 + w2[:, None] * p2
        nsurf = w0[:, None] * n0 + w1[:, None] * n1 + w2[:, None] * n2
        nsurf = F.normalize(nsurf, dim=-1)

        return psurf + d * nsurf

    def barycentric_2d(self, p, a, b, c):
        """
        p,a,b,c: (...,2)
        return w0,w1,w2: (...,)
        可微（对 p 可微）
        """
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        d20 = (v2 * v0).sum(-1)
        d21 = (v2 * v1).sum(-1)
        denom = d00 * d11 - d01 * d01
        denom = denom.clamp_min(1e-12)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    def clamp_small_negatives(self, w0, w1, w2):
        """
        """
        w = torch.stack([w0, w1, w2], dim=-1)
        w = torch.where(w < -1e-4, w, torch.clamp(w, min=0.0))
        s = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        w = w / s
        return w[..., 0], w[..., 1], w[..., 2]

    def compute_vertex_normals(self, verts, faces):
        """
        verts: (V,3), faces: (F,3)
        return vnormals: (V,3)
        """
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F,3) area-weighted
        vnorm = torch.zeros_like(verts)
        vnorm.index_add_(0, faces[:, 0], fn)
        vnorm.index_add_(0, faces[:, 1], fn)
        vnorm.index_add_(0, faces[:, 2], fn)
        vnorm = F.normalize(vnorm, dim=-1)
        return vnorm

    def _flame_verts_and_normals(self):
        flame_output = self.model(
            betas=self.get_shape,
            neck_pose=self.get_neck_pose,
            expression=self.get_expression,
            jaw_pose=self.get_jaw_pose,
            leye_pose=self.get_leye_pose,
            reye_pose=self.get_reye_pose,
            return_verts=True
        )
        verts = flame_output.vertices.squeeze()
        verts = (verts - self.center) * self.scale
        verts[:, [1, 2]] = verts[:, [2, 1]]
        verts *= 1.1 ** (-self.flame_scale)

        faces = self.get_faces.long()
        vnormals = self.compute_vertex_normals(verts, faces)
        return verts, vnormals

    def _face_tangent_bitangent_from_uv(self, verts):
        faces = self.get_faces.long()  # (F,3) geom indices
        ft = self.ft.long()  # (F,3) uv indices
        vt = self.vt  # (T,2)

        # 确保 faces 和 ft 使用相同数量的面
        num_faces = min(faces.shape[0], ft.shape[0])
        faces = faces[:num_faces]
        ft = ft[:num_faces]

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        uv0 = vt[ft[:, 0]]
        uv1 = vt[ft[:, 1]]
        uv2 = vt[ft[:, 2]]

        dp1 = v1 - v0
        dp2 = v2 - v0
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0

        r = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]  # (F,)
        bad = r.abs() < 1e-8
        # r = r.clamp_min(1e-10)
        sign = torch.where(r >= 0, torch.ones_like(r), -torch.ones_like(r))
        r = torch.where(bad, sign * 1e-10, r)
        inv = 1.0 / r

        Tu = (dp1 * duv2[:, 1:2] - dp2 * duv1[:, 1:2]) * inv[:, None]  # ∂p/∂u
        Tv = (-dp1 * duv2[:, 0:1] + dp2 * duv1[:, 0:1]) * inv[:, None]  # ∂p/∂v

        cap = 50.0  #
        bad = bad | (Tu.norm(dim=-1) > cap) | (Tv.norm(dim=-1) > cap)
        # fallback：UV 退化时用几何构造的切线（不带 uv 缩放，但避免爆炸）
        if bad.any():
            tris = torch.stack([v0, v1, v2], dim=1)
            R = self.tbn(tris)  # (F,3,3) 里面是 X,Y,Z
            Tu[bad] = R[bad][:, 0, :]
            Tv[bad] = R[bad][:, 1, :]

        return Tu, Tv

    def _gaussian_jacobian_J(self, verts, vnormals):
        """Jacobian of F at each Gaussian's (u,v,d).

        TeGA uses J_F to map Σ_uvd -> Σ_xyz: Σ_xyz = J_F Σ_uvd J_F^T.
        We provide two modes:
          - 'analytic': fast approximation using per-face tangents and interpolated normal.
          - 'autodiff': compute J_F by differentiating F w.r.t (u,v,d) (closer to paper).
        """
        mode = getattr(self, "jacobian_mode", "autodiff")
        if mode == "analytic":
            return self._gaussian_jacobian_J_analytic(verts, vnormals)
        elif mode == "autodiff":
            return self._gaussian_jacobian_J_autodiff(verts, vnormals)
        else:
            raise ValueError(f"Unknown jacobian_mode={mode}. Use 'analytic' or 'autodiff'.")

    def _gaussian_jacobian_J_analytic(self, verts, vnormals):
        """Analytic (piecewise-linear) Jacobian using UV-derived tangents.
        Note: This ignores the derivative of the normalized normal term w.r.t (u,v) when d!=0,
        but is fast and stable.
        """
        f = self._face_idx.long()
        faces = self.get_faces.long()
        num_faces = min(faces.shape[0], self.ft.shape[0])
        f = f.clamp(0, num_faces - 1)

        # Per-face tangents in world-space: ∂p/∂u, ∂p/∂v
        Tu_face, Tv_face = self._face_tangent_bitangent_from_uv(verts)  # (F,3), (F,3)
        Tu = Tu_face[f]
        Tv = Tv_face[f]

        # Interpolated normal at (u,v): ∂F/∂d
        gtri = faces[f]
        n0 = vnormals[gtri[:, 0]]
        n1 = vnormals[gtri[:, 1]]
        n2 = vnormals[gtri[:, 2]]

        ttri = self.ft[f]
        uv0 = self.vt[ttri[:, 0]]
        uv1 = self.vt[ttri[:, 1]]
        uv2 = self.vt[ttri[:, 2]]
        uv = self.get_uv

        w0, w1, w2 = self.barycentric_2d(uv, uv0, uv1, uv2)
        w0, w1, w2 = self.clamp_small_negatives(w0, w1, w2)
        n = w0[:, None] * n0 + w1[:, None] * n1 + w2[:, None] * n2
        n = F.normalize(n, dim=-1)

        return torch.stack([Tu, Tv, n], dim=-1)  # (N,3,3) columns=[du,dv,dd]

    def _gaussian_jacobian_J_autodiff(self, verts, vnormals):
        """Autodiff Jacobian of F w.r.t (u,v,d).
        This matches TeGA's recommendation to capture the full local transformation.
        By default we do NOT keep higher-order graph (jacobian_create_graph=False) for efficiency.
        """
        with torch.enable_grad():
            create_graph = getattr(self, "jacobian_create_graph", False)

            # Build (u,v,d) tensor. Keep graph if available (training); otherwise detach for eval.
            uvd = torch.cat([self.get_uv, self._d], dim=1)
            if not uvd.requires_grad:
                uvd = uvd.detach().requires_grad_(True)

            xyz = self._map_uvd_to_xyz(uvd, verts, vnormals)

            # Per-point Jacobian: J[n, out_dim, in_dim]
            grads_rows = []
            for c in range(3):
                grad_outputs = torch.zeros_like(xyz)
                grad_outputs[:, c] = 1.0
                g = torch.autograd.grad(
                    outputs=xyz,
                    inputs=uvd,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=create_graph,
                    only_inputs=True,
                    allow_unused=False,
                )[0]  # (N,3)
                grads_rows.append(g)
            J = torch.stack(grads_rows, dim=1)  # (N,3,3) rows=xyz dims, cols=uvd dims
        return J

    def get_world_scale_max_approx(self):
        verts, vnormals = self._flame_verts_and_normals()
        J = self._gaussian_jacobian_J(verts, vnormals)  # (N,3,3)
        Tu = J[:, :, 0]
        Tv = J[:, :, 1]
        su, sv, sd = self.get_scaling[:, 0], self.get_scaling[:, 1], self.get_scaling[:, 2]
        sw = torch.stack([su * Tu.norm(dim=-1), sv * Tv.norm(dim=-1), sd], dim=-1)
        return sw.max(dim=-1).values  # (N,)

    def _safe_sigmoid_uv(self, uv_logits, eps=1e-6):
        # 防止 inverse_sigmoid 在 0/1 出现 inf
        uv = torch.sigmoid(uv_logits)
        return uv.clamp(eps, 1.0 - eps)

    @property
    def get_uv(self):
        return self._safe_sigmoid_uv(self._uv_logits)

    @torch.no_grad()
    def clamp_uv_logits_(self, m=10.0):
        # 可选：避免 logits 过大导致 sigmoid 饱和
        self._uv_logits.clamp_(-m, m)

    def _build_uv_grid(self, res=256, maxK=32):
        """
        把 UV 平面划成 res×res 网格。每个 cell 存最多 maxK 个候选 face。
        UV atlas 9976 faces 量级，res=256 通常够用且内存不爆。
        """
        vt = self.vt.detach().cpu().numpy()
        ft = self.ft.detach().cpu().numpy()
        F = ft.shape[0]

        # 每个 face 的 uv 三角形
        uvs = vt[ft]  # (F,3,2)
        umin = uvs[:, :, 0].min(1)
        umax = uvs[:, :, 0].max(1)
        vmin = uvs[:, :, 1].min(1)
        vmax = uvs[:, :, 1].max(1)

        # face 覆盖哪些 cell
        def to_cell(x):
            return np.clip((x * res).astype(np.int32), 0, res - 1)

        i0 = to_cell(umin)
        i1 = to_cell(umax)
        j0 = to_cell(vmin)
        j1 = to_cell(vmax)

        cells = [[] for _ in range(res * res)]
        for f in range(F):
            for j in range(j0[f], j1[f] + 1):
                base = j * res
                for i in range(i0[f], i1[f] + 1):
                    cells[base + i].append(f)

        # pad 成 (res*res, maxK)
        grid = -np.ones((res * res, maxK), dtype=np.int32)
        for c in range(res * res):
            lst = cells[c][:maxK]
            if len(lst) > 0:
                grid[c, :len(lst)] = np.array(lst, dtype=np.int32)

        self.uv_grid_res = res
        self.uv_grid = torch.from_numpy(grid).to(self.device)  # long/int 都行，下面会转 long

    def barycentric_2d_batch(self, p, a, b, c):
        """
        p: (N,1,2) 或 (N,K,2)
        a,b,c: (N,K,2)
        return w0,w1,w2: (N,K)
        """
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        d20 = (v2 * v0).sum(-1)
        d21 = (v2 * v1).sum(-1)
        denom = (d00 * d11 - d01 * d01).clamp_min(1e-12)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    @torch.no_grad()
    def update_face_idx_from_uv(self, eps=1e-6, eps_keep=1e-4, mask=None, return_stats=False):
        """
        Sticky 更新：
        - 如果 uv 仍在 old face 内（允许 eps_keep），则保持 old face，不做任何替换。
        - 只有 uv 明确不在 old face 内，才从 uv_grid 候选中寻找新 face。
        - 没命中任何候选：保持 old face。
        支持 mask：只更新部分点（True=更新）。
        """
        # 0) 准备
        if not hasattr(self, "uv_grid") or self.uv_grid is None:
            raise RuntimeError("uv_grid not built. Call _build_uv_grid() first.")
        uv = self.get_uv  # (N,2) in [0,1]
        old_all = self._face_idx.long()
        N = uv.shape[0]
        res = self.uv_grid_res

        if mask is None:
            sel = torch.ones(N, device=self.device, dtype=torch.bool)
        else:
            sel = mask.to(device=self.device, dtype=torch.bool)
            if sel.numel() != N:
                raise ValueError(f"mask shape mismatch: mask has {sel.numel()} elems, N={N}")

        if not sel.any():
            return

        uv_sel = uv[sel]  # (M,2)
        old_sel = old_all[sel]  # (M,)

        M = uv_sel.shape[0]

        # 1) 先检查 old face 是否仍包含 uv（粘性：inside 就不换）
        ttri_old = self.ft[old_sel]  # (M,3)
        tri_uv_old = self.vt[ttri_old]  # (M,3,2)
        a0 = tri_uv_old[:, 0, :]
        b0 = tri_uv_old[:, 1, :]
        c0 = tri_uv_old[:, 2, :]

        # barycentric_2d_batch 期望 (M,K,2)，这里 K=1
        p0 = uv_sel[:, None, :]  # (M,1,2)
        w0, w1, w2 = self.barycentric_2d_batch(p0, a0[:, None, :], b0[:, None, :], c0[:, None, :])
        w0 = w0[:, 0]
        w1 = w1[:, 0]
        w2 = w2[:, 0]

        inside_old = (w0 >= -eps_keep) & (w1 >= -eps_keep) & (w2 >= -eps_keep)

        # 需要重新找 face 的点（明确不在 old face 内）
        need = ~inside_old
        bad_num = ~(torch.isfinite(w0) & torch.isfinite(w1) & torch.isfinite(w2))
        need = need | bad_num

        need_cnt = int(need.sum().item())

        if not need.any():
            if return_stats:
                return {"need": 0, "updated": 0, "no_hit": 0}
            return

        uv_need = uv_sel[need]    # (M2,2)
        old_need = old_sel[need]  # (M2, )
        M2 = uv_need.shape[0]

        # 2) 对 need 子集：按网格候选找新 face（你原来的逻辑）
        ij = (uv_need * res).long().clamp(0, res - 1)  # (M2,2)
        cell = ij[:, 1] * res + ij[:, 0]  # (M2,)

        cands = self.uv_grid[cell].long()  # (M2,K)
        valid = cands >= 0

        cands_clamped = cands.clamp_min(0)
        ttri = self.ft[cands_clamped]  # (M2,K,3)
        tri_uv = self.vt[ttri]  # (M2,K,3,2)

        a = tri_uv[:, :, 0, :]
        b = tri_uv[:, :, 1, :]
        c = tri_uv[:, :, 2, :]

        p = uv_need[:, None, :]  # (M2,1,2) -> broadcast to (M2,K,2)
        ww0, ww1, ww2 = self.barycentric_2d_batch(p, a, b, c)

        inside = valid & (ww0 >= -eps) & (ww1 >= -eps) & (ww2 >= -eps)

        # 选“最内侧”的三角形：min(w) 最大
        score = torch.minimum(torch.minimum(ww0, ww1), ww2)
        score = torch.where(inside, score, torch.full_like(score, -1e9))
        best = score.argmax(dim=1)  # (M2,)

        new_face = cands[torch.arange(M2, device=self.device), best]  # (M2,)

        # fallback：没命中任何候选，保持 old face
        no_hit = score.max(dim=1).values < -1e8
        new_face[no_hit] = old_need[no_hit]

        # 计算本轮 need 子集在“全局高斯索引”上的位置
        all_idx = torch.arange(N, device=self.device)
        sel_idx = all_idx[sel]  # (M,)
        need_idx = sel_idx[need]  # (M2,)  <-- uv_need/old_need/new_face 对应的全局索引

        changed = (~no_hit) & (new_face != old_need)
        if changed.any():
            ch_local = torch.nonzero(changed, as_tuple=False).squeeze(-1)  # (Mchg,)  in [0,M2)
            ch_idx = need_idx[ch_local]  # (Mchg,) global gaussian indices
            f_old = old_need[ch_local]  # (Mchg,)
            f_new = new_face[ch_local]  # (Mchg,)
            uv_ch = uv_need[ch_local]  # (Mchg,2)
            scaling_backup = self._scaling.data[ch_idx].clone()
            rot_backup = self._rotation.data[ch_idx].clone()
            f_backup = f_old.clone()  # changed 子集的 old face

            # 1) mesh + per-face tangents（一次性算）
            verts, vnormals = self._flame_verts_and_normals()
            Tu_face, Tv_face = self._face_tangent_bitangent_from_uv(verts)  # (F,3),(F,3)
            faces_geo = self.get_faces.long()
            num_faces = min(faces_geo.shape[0], self.ft.shape[0])

            def jacobian_autodiff_subset(uv, d, face_idx, verts, vnormals):
                # uv: (B,2), d: (B,1), face_idx: (B,)
                with torch.enable_grad():
                    uvd = torch.cat([uv, d], dim=1).detach().requires_grad_(True)  # (B,3)

                    xyz = self._map_uvd_to_xyz(uvd, verts, vnormals, face_idx=face_idx)  # (B,3)

                    rows = []
                    for c in range(3):
                        grad_outputs = torch.zeros_like(xyz)
                        grad_outputs[:, c] = 1.0
                        g = torch.autograd.grad(
                            outputs=xyz,
                            inputs=uvd,
                            grad_outputs=grad_outputs,
                            create_graph=False,
                            retain_graph=True,
                            only_inputs=True
                        )[0]  # (B,3)  d xyz_c / d uvd
                        rows.append(g)

                    J = torch.stack(rows, dim=1)  # (B,3,3)  rows=out_dim, cols=in_dim
                    return J

            # 3) 取当前这批点的 UVD 参数（旧参数）
            s_old = self.get_scaling[ch_idx]  # (B,3)  positive
            q_old = self.rotation_activation(self._rotation[ch_idx])  # (B,4)
            R_old = build_rotation(q_old)  # (B,3,3)

            # 4) J_old/J_new
            uv_ch = uv_need[ch_local]  # (B,2)
            d_ch = self._d[need_idx[ch_local]]  # (B,1)  用当前 d
            J_old = jacobian_autodiff_subset(uv_ch, d_ch, f_old, verts, vnormals)
            J_new = jacobian_autodiff_subset(uv_ch, d_ch, f_new, verts, vnormals)

            # 5) Σ_uvd_old -> Σ_xyz_target
            # 用 AA^T 更稳：A = R * diag(s)
            A_old = R_old * (s_old[:, None, :])  # (B,3,3)
            Sigma_uvd_old = A_old @ A_old.transpose(1, 2)
            Sigma_xyz = J_old @ Sigma_uvd_old @ J_old.transpose(1, 2)
            sigma_floor = 1e-4  # 与渲染用的保持一致
            Sigma_xyz = Sigma_xyz + torch.eye(3, device=Sigma_xyz.device)[None] * (sigma_floor ** 2)

            # 6) 反解 Σ_uvd_new = J_new^{-1} Σ_xyz J_new^{-T}
            U, S, Vh = torch.linalg.svd(J_new)  # J_new: (B,3,3), S: (B,3)
            # 关键：给奇异值加下限，避免病态爆炸（数值可调）
            s_floor = 1e-4  # 先试 1e-4；还有线就 1e-3
            S_clamp = S.clamp_min(s_floor)

            # 可选：如果条件数太大，直接跳过 C2（或退回 C1）
            cond = S[:, 0] / S[:, 2].clamp_min(1e-12)
            badJ = cond > 1e3  # 阈值可调：1e3~1e4

            # 逆：J^{-1} = V diag(1/S) U^T
            V = Vh.transpose(-2, -1)
            Jinv = (V * (1.0 / S_clamp)[:, None, :]) @ U.transpose(-2, -1)
            Sigma_uvd_new = Jinv @ Sigma_xyz @ Jinv.transpose(1, 2)
            Sigma_uvd_new = 0.5 * (Sigma_uvd_new + Sigma_uvd_new.transpose(1, 2))  # 强制对称

            # 7) Σ_uvd_new -> (R_new, s_new) via eigen
            evals, evecs = torch.linalg.eigh(Sigma_uvd_new)  # evals(B,3) asc
            evals = torch.clamp(evals, min=1e-10)  # 防止负/0

            ratio_max = 100.0
            e_max = evals[:, 2]
            e_min = evals[:, 0].clamp_min(1e-10)
            too_aniso = (e_max / e_min) > (ratio_max ** 2)
            if too_aniso.any():
                # 抬高最小特征值，使比值不超过 ratio_max
                evals[too_aniso, 0] = e_max[too_aniso] / (ratio_max ** 2)

            s_new = torch.sqrt(evals)  # (B,3)
            R_new = evecs  # (B,3,3)

            # eigh 的 evecs 可能 det=-1，修正为 proper rotation
            det = torch.det(R_new)
            neg = det < 0
            if neg.any():
                R_new[neg, :, 2] *= -1.0

            # 8) rotation matrix -> quaternion (w,x,y,z)  注意：需与你的 build_rotation 一致
            def rotmat_to_quat_wxyz(R):
                # R: (B,3,3)
                B = R.shape[0]
                q = torch.empty((B, 4), device=R.device, dtype=R.dtype)

                tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
                m0 = tr > 0

                # case 0: trace > 0
                if m0.any():
                    t = tr[m0]
                    S = torch.sqrt(t + 1.0) * 2.0
                    q[m0, 0] = 0.25 * S
                    q[m0, 1] = (R[m0, 2, 1] - R[m0, 1, 2]) / S
                    q[m0, 2] = (R[m0, 0, 2] - R[m0, 2, 0]) / S
                    q[m0, 3] = (R[m0, 1, 0] - R[m0, 0, 1]) / S

                # case 1/2/3: pick largest diagonal
                m1 = (~m0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
                m2 = (~m0) & (~m1) & (R[:, 1, 1] > R[:, 2, 2])
                m3 = (~m0) & (~m1) & (~m2)

                if m1.any():
                    S = torch.sqrt(1.0 + R[m1, 0, 0] - R[m1, 1, 1] - R[m1, 2, 2]) * 2.0
                    q[m1, 0] = (R[m1, 2, 1] - R[m1, 1, 2]) / S
                    q[m1, 1] = 0.25 * S
                    q[m1, 2] = (R[m1, 0, 1] + R[m1, 1, 0]) / S
                    q[m1, 3] = (R[m1, 0, 2] + R[m1, 2, 0]) / S

                if m2.any():
                    S = torch.sqrt(1.0 + R[m2, 1, 1] - R[m2, 0, 0] - R[m2, 2, 2]) * 2.0
                    q[m2, 0] = (R[m2, 0, 2] - R[m2, 2, 0]) / S
                    q[m2, 1] = (R[m2, 0, 1] + R[m2, 1, 0]) / S
                    q[m2, 2] = 0.25 * S
                    q[m2, 3] = (R[m2, 1, 2] + R[m2, 2, 1]) / S

                if m3.any():
                    S = torch.sqrt(1.0 + R[m3, 2, 2] - R[m3, 0, 0] - R[m3, 1, 1]) * 2.0
                    q[m3, 0] = (R[m3, 1, 0] - R[m3, 0, 1]) / S
                    q[m3, 1] = (R[m3, 0, 2] + R[m3, 2, 0]) / S
                    q[m3, 2] = (R[m3, 1, 2] + R[m3, 2, 1]) / S
                    q[m3, 3] = 0.25 * S

                q = torch.nn.functional.normalize(q, dim=-1)
                return q

            q_new = rotmat_to_quat_wxyz(R_new)

            # 9) 写回参数（log-scale + quat）
            # scaling_activation里你已经 clamp(min=1e-5)，这里也 clamp 一下避免 log 出问题
            s_new = torch.clamp(s_new, min=1e-5)
            self._scaling.data[ch_idx] = torch.log(s_new)
            self._rotation.data[ch_idx] = q_new
            # 1) 用“写回后的参数”重建 Σ_uvd_chk
            s_chk = self.get_scaling[ch_idx]  # exp后的正尺度
            q_chk = self.rotation_activation(self._rotation[ch_idx])
            R_chk = build_rotation(q_chk)
            A_chk = R_chk * (s_chk[:, None, :])
            Sigma_uvd_chk = A_chk @ A_chk.transpose(1, 2)

            # 2) 与你计算的 Sigma_uvd_new 做一致性检查
            num = (Sigma_uvd_chk - Sigma_uvd_new).abs().amax(dim=(1, 2))
            den = Sigma_uvd_new.abs().amax(dim=(1, 2)).clamp_min(1e-12)
            rel = num / den

            # 3) 额外安全条件：尺度/各向异性上限（防“线条高斯”）
            ratio = (s_chk.max(dim=-1).values / s_chk.min(dim=-1).values.clamp_min(1e-12))
            bad = (~torch.isfinite(rel)) | (rel > 1e-2) | (~torch.isfinite(ratio)) | (ratio > 100.0)

            if bad.any():
                bad_idx = ch_idx[bad]
                self._scaling.data[bad_idx] = scaling_backup[bad]
                self._rotation.data[bad_idx] = rot_backup[bad]
                # 关键：取消这些点的 face 变更
                # ch_local 是 changed 子集在 need 子集内的局部索引
                new_face[ch_local[bad]] = f_backup[bad]

        updated_cnt = int((new_face != old_need).sum().item())
        no_hit_cnt = int(no_hit.sum().item())

        # clamp 合法范围
        num_faces = self.ft.shape[0]
        new_face = new_face.clamp(0, num_faces - 1)

        # 3) 写回：only replace for need subset
        old_sel_out = old_sel.clone()
        old_sel_out[need] = new_face

        old_all_out = old_all.clone()
        old_all_out[sel] = old_sel_out

        self._face_idx = old_all_out

        if return_stats:
            return {"need": need_cnt, "updated": updated_cnt, "no_hit": no_hit_cnt}

    def sample_uv_feat(self, uv):
        # uv: (N,2) in [0,1]
        feat_map = self._uv_atlas_map()
        grid = uv * 2 - 1  # -> [-1,1]
        grid = grid.view(1, -1, 1, 2)  # (1,N,1,2)
        feat = F.grid_sample(feat_map, grid, mode="bilinear", padding_mode="border", align_corners=False)
        feat = feat.view(self.uv_feat_dim, -1).T  # (N,C)
        return feat

    @property
    def get_features(self):
        """
        覆盖 base GaussianModel 的 get_features。
        输出 shape 要和原来一致： (N, (sh_degree+1)^2, 3)
        你现在 sh_degree=0 => (N,1,3)
        """
        if (not getattr(self, "use_uv_atlas", False)):
            return torch.cat((self._features_dc, self._features_rest), dim=1)

        # 仅支持 sh_degree=0 （你当前就是0）
        if self.max_sh_degree != 0:
            # fallback：只替换 DC，rest 仍用逐点
            uv = self.get_uv
            feat = self.sample_uv_feat(uv)
            rgb, _ = self.uv_heads(feat)  # (N,3)
            sh0 = RGB2SH(rgb)  # (N,3)
            f_dc = sh0[:, None, :]  # (N,1,3)
            return torch.cat((f_dc, self._features_rest), dim=1)

        uv = self.get_uv
        feat = self.sample_uv_feat(uv)
        if torch.isnan(feat).any():
            print("[NaN] feat is NaN")
        rgb, _ = self.uv_heads(feat)  # (N,3) in [0,1]
        sh0 = RGB2SH(rgb)  # (N,3)
        f_dc = sh0[:, None, :]  # (N,1,3)
        return f_dc

    @property
    def get_opacity(self):
        """
        覆盖 base GaussianModel 的 get_opacity（返回已 sigmoid 的 alpha）
        """
        if (not getattr(self, "use_uv_atlas", False)):
            return self.opacity_activation(self._opacity)

        uv = self.get_uv
        feat = self.sample_uv_feat(uv)
        if torch.isnan(feat).any():
            print("[NaN] feat is NaN")
        _, op = self.uv_heads(feat)  # op: (N,1) in [0,1]
        return op

    def create_from_flame(self, spatial_lr_scale: float, flame_scale: float, N=100000):
        self.spatial_lr_scale = spatial_lr_scale

        # flame
        flame_model = self.model
        # flame: shape and expression
        betas = torch.zeros([1, self.num_betas], dtype=torch.float32, device=self.device)
        expression = torch.zeros([1, self.num_expression], dtype=torch.float32, device=self.device)
        # flame: triangles
        flame_output = flame_model(betas=betas, expression=expression, return_verts=True)
        vertices = flame_output.vertices.squeeze()
        faces = torch.tensor(flame_model.faces.astype(np.int64), dtype=torch.long, device=self.device)
        # rescale and recenter
        vmin = vertices.min(0)[0]
        vmax = vertices.max(0)[0]
        ori_center = (vmin + vmax) / 2
        ori_scale = 0.6 / (vmax - vmin).max()
        vertices = (vertices - ori_center) * ori_scale
        # coordinate system: opengl --> blender (switch y/z)
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        vertices *= 1.1 ** (-flame_scale)

        self.flame_scale = flame_scale
        self.center = ori_center.detach()
        self.scale = ori_scale.detach()
        self._shape = nn.Parameter(betas.contiguous().requires_grad_(True))
        self._expression = expression.detach()
        self._faces = faces

        # 3DGS
        mesh = trimesh.Trimesh(vertices.detach().cpu(), faces.detach().cpu())
        samples, face_index = trimesh.sample.sample_surface(mesh, N)
        samples = torch.from_numpy(np.asarray(samples)).float().to(self.device)  # (N,3)
        face_index = torch.from_numpy(face_index.astype(np.int64)).to(self.device)  # (N,)
        self.num_gs = N
        self._face_idx = face_index
        # 计算 barycentric（在 3D 三角形里）
        gtri = faces[face_index].long()  # (N,3)
        p0 = vertices[gtri[:, 0]]
        p1 = vertices[gtri[:, 1]]
        p2 = vertices[gtri[:, 2]]
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = samples - p0
        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        d20 = (v2 * v0).sum(-1)
        d21 = (v2 * v1).sum(-1)
        den = (d00 * d11 - d01 * d01).clamp_min(1e-12)
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        w0 = 1.0 - w1 - w2
        # 用 barycentric 插值 UV（注意：用 ft 索引 vt）
        ttri = self.ft[face_index]  # (N,3)
        uv0 = self.vt[ttri[:, 0]]
        uv1 = self.vt[ttri[:, 1]]
        uv2 = self.vt[ttri[:, 2]]
        uv = w0[:, None] * uv0 + w1[:, None] * uv1 + w2[:, None] * uv2  # (N,2)
        # 初始化 d=0
        d = torch.zeros((N, 1), device=self.device)
        self._d = nn.Parameter(d.requires_grad_(True))
        print("Number of points at initialisation : ", uv.shape[0])
        eps = 1e-6
        uv = uv.clamp(eps, 1.0 - eps)
        uv_logits = inverse_sigmoid(uv)  # (N,2)
        self._uv_logits = nn.Parameter(uv_logits.requires_grad_(True))

        fused_color = torch.ones((N,3), dtype=torch.float32, device=self.device) * 0.5
        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001)
        r_world = torch.sqrt(dist2)

        with torch.no_grad():
            vnormals = self.compute_vertex_normals(vertices, self._faces)

        J = self._gaussian_jacobian_J(vertices, vnormals)  # (N,3,3)
        Tu = J[:, :, 0]
        Tv = J[:, :, 1]

        tu_norm = Tu.norm(dim=-1).clamp_min(1e-6)
        tv_norm = Tv.norm(dim=-1).clamp_min(1e-6)
        s_u = (r_world / tu_norm)
        s_v = (r_world / tv_norm)
        s_d = r_world

        scales_uvd = torch.stack([s_u, s_v, s_d], dim=-1)  # (N,3)
        scales_log = torch.log(scales_uvd)

        self._scaling = nn.Parameter(scales_log.requires_grad_(True))

        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((N, 1), dtype=torch.float, device="cuda"))

        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.num_gs), device="cuda")

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._faces.shape[1]):
            l.append('face_{}'.format(i))
        return l

    def _vertex_dtype_uv(self):
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('u_logit', 'f4'), ('v_logit', 'f4'),
            ('d', 'f4'),
            ('face_idx', 'i4'),
            ('opacity', 'f4'),
        ]

        # f_dc_0..2
        for i in range(3):
            dtype.append((f'f_dc_{i}', 'f4'))

        # f_rest_0..(3*(sh^2)-3-1)
        n_rest = 3 * (self.max_sh_degree + 1) ** 2 - 3
        for i in range(n_rest):
            dtype.append((f'f_rest_{i}', 'f4'))

        # scale_*
        for i in range(self._scaling.shape[1]):
            dtype.append((f'scale_{i}', 'f4'))

        # rot_*
        for i in range(self._rotation.shape[1]):
            dtype.append((f'rot_{i}', 'f4'))

        return dtype

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        # 用当前渲染坐标写入 ply（仅用于可视化）
        xyz = self.get_xyz.detach().cpu().numpy().astype(np.float32)
        normals = np.zeros_like(xyz, dtype=np.float32)

        uv_logits = self._uv_logits.detach().cpu().numpy().astype(np.float32)  # (N,2)
        d = self._d.detach().cpu().numpy().astype(np.float32)  # (N,1)
        face_idx = self._face_idx.detach().cpu().numpy().astype(np.int32)  # (N,)

        # opacities = self._opacity.detach().cpu().numpy().astype(np.float32)  # (N,1)
        scales = self._scaling.detach().cpu().numpy().astype(np.float32)  # (N,3)
        rots = self._rotation.detach().cpu().numpy().astype(np.float32)  # (N,4)
        with torch.no_grad():
            # 1) opacity：get_opacity 是 sigmoid 后的值，写 ply 需要 logit（与你的 load_ply 对齐）
            op = self.get_opacity.detach()
            if op.ndim == 1:
                op = op[:, None]
            op = op.clamp(1e-6, 1.0 - 1e-6)
            op_logits = self.inverse_opacity_activation(op)  # -> 与 self._opacity 同语义
            opacities = op_logits.cpu().numpy().astype(np.float32)  # (N,1)

            # 2) features：拿 get_features，然后拆成 dc/rest 写入字段
            feats = self.get_features.detach()  # 期望 (N, S, 3) 或 (N,3,S) 或 (N,3*S)

            if feats.ndim == 3:
                # (N,S,3) 或 (N,3,S)
                if feats.shape[-1] == 3:
                    feats_s3 = feats
                elif feats.shape[1] == 3:
                    feats_s3 = feats.permute(0, 2, 1).contiguous()
                else:
                    raise RuntimeError(f"Unexpected get_features shape: {tuple(feats.shape)}")
            elif feats.ndim == 2:
                # (N, 3*S)
                if feats.shape[1] % 3 != 0:
                    raise RuntimeError(f"Unexpected get_features shape: {tuple(feats.shape)}")
                S = feats.shape[1] // 3
                feats_s3 = feats.view(feats.shape[0], S, 3).contiguous()
            else:
                raise RuntimeError(f"Unexpected get_features shape: {tuple(feats.shape)}")

            # feats_s3: (N, S, 3)
            N = feats_s3.shape[0]
            f_dc = feats_s3[:, 0:1, :].transpose(1, 2).reshape(N, -1).cpu().numpy().astype(np.float32)  # (N,3)
            f_rest = feats_s3[:, 1:, :].transpose(1, 2).reshape(N, -1).cpu().numpy().astype(
                np.float32)  # (N, 3*(S-1))

            # 3) 如果 max_sh_degree 期望的 rest 通道数与当前不一致：补零/裁剪，避免 ply 字段对不上
            expected_rest = 3 * (self.max_sh_degree + 1) ** 2 - 3
            if f_rest.shape[1] != expected_rest:
                if f_rest.shape[1] < expected_rest:
                    pad = np.zeros((N, expected_rest - f_rest.shape[1]), dtype=np.float32)
                    f_rest = np.concatenate([f_rest, pad], axis=1)
                else:
                    f_rest = f_rest[:, :expected_rest]

        # features
        # f_dc = self._features_dc.detach().transpose(1, 2).contiguous().cpu().numpy().astype(np.float32)
        # f_dc = f_dc.reshape(f_dc.shape[0], -1)  # (N,3)

        # f_rest = self._features_rest.detach().transpose(1, 2).contiguous().cpu().numpy().astype(np.float32)
        # f_rest = f_rest.reshape(f_rest.shape[0], -1)  # (N, 3*((sh^2)-1))

        # 组装 vertex element
        N = xyz.shape[0]
        vtx = np.empty(N, dtype=self._vertex_dtype_uv())

        vtx['x'] = xyz[:, 0]
        vtx['y'] = xyz[:, 1]
        vtx['z'] = xyz[:, 2]
        vtx['nx'] = normals[:, 0]
        vtx['ny'] = normals[:, 1]
        vtx['nz'] = normals[:, 2]

        vtx['u_logit'] = uv_logits[:, 0]
        vtx['v_logit'] = uv_logits[:, 1]
        vtx['d'] = d[:, 0]
        vtx['face_idx'] = face_idx
        vtx['opacity'] = opacities[:, 0]

        # f_dc_0..2
        for i in range(3):
            vtx[f'f_dc_{i}'] = f_dc[:, i]

        # f_rest_*
        for i in range(f_rest.shape[1]):
            vtx[f'f_rest_{i}'] = f_rest[:, i]

        # scale_*
        for i in range(scales.shape[1]):
            vtx[f'scale_{i}'] = scales[:, i]

        # rot_*
        for i in range(rots.shape[1]):
            vtx[f'rot_{i}'] = rots[:, i]

        el_v = PlyElement.describe(vtx, 'vertex')

        # 可选：保存 shape
        elements = [el_v]
        if isinstance(self._shape, torch.nn.Parameter) and self._shape.numel() > 0:
            shape = self._shape.detach().cpu().numpy().astype(np.float32)  # (1, num_betas)
            shape_dtype = [(f'shape_{i}', 'f4') for i in range(shape.shape[1])]
            shape_el = np.empty(shape.shape[0], dtype=shape_dtype)
            for i in range(shape.shape[1]):
                shape_el[f'shape_{i}'] = shape[:, i]
            elements.append(PlyElement.describe(shape_el, 'shape'))

        PlyData(elements).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        v = plydata.elements[0]  # vertex element

        # 必要字段检查
        required = ["u_logit", "v_logit", "d", "face_idx", "opacity"]
        for k in required:
            if k not in v.data.dtype.names:
                raise KeyError(f"PLY missing required field '{k}'. This loader expects uv_logits+d+face_idx format.")

        uv_logits = np.stack([np.asarray(v["u_logit"]), np.asarray(v["v_logit"])], axis=1).astype(np.float32)  # (N,2)
        d = np.asarray(v["d"]).astype(np.float32)[:, None]  # (N,1)
        face_idx = np.asarray(v["face_idx"]).astype(np.int64)  # (N,)
        opacities = np.asarray(v["opacity"]).astype(np.float32)[:, None]  # (N,1)

        # features_dc
        f_dc = np.zeros((uv_logits.shape[0], 3, 1), dtype=np.float32)
        f_dc[:, 0, 0] = np.asarray(v["f_dc_0"]).astype(np.float32)
        f_dc[:, 1, 0] = np.asarray(v["f_dc_1"]).astype(np.float32)
        f_dc[:, 2, 0] = np.asarray(v["f_dc_2"]).astype(np.float32)

        # features_rest
        extra_f_names = [p.name for p in v.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        expected = 3 * (self.max_sh_degree + 1) ** 2 - 3
        if len(extra_f_names) != expected:
            raise ValueError(f"Unexpected f_rest count: got {len(extra_f_names)}, expected {expected}.")
        f_rest = np.zeros((uv_logits.shape[0], len(extra_f_names)), dtype=np.float32)
        for i, name in enumerate(extra_f_names):
            f_rest[:, i] = np.asarray(v[name]).astype(np.float32)
        f_rest = f_rest.reshape((uv_logits.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # scaling
        scale_names = [p.name for p in v.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((uv_logits.shape[0], len(scale_names)), dtype=np.float32)
        for i, name in enumerate(scale_names):
            scales[:, i] = np.asarray(v[name]).astype(np.float32)

        # rotation
        rot_names = [p.name for p in v.properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((uv_logits.shape[0], len(rot_names)), dtype=np.float32)
        for i, name in enumerate(rot_names):
            rots[:, i] = np.asarray(v[name]).astype(np.float32)

        device = "cuda"

        # 位置参数：uv_logits + d
        self._uv_logits = nn.Parameter(torch.tensor(uv_logits, dtype=torch.float32, device=device).requires_grad_(True))
        self._d = nn.Parameter(torch.tensor(d, dtype=torch.float32, device=device).requires_grad_(True))
        self._face_idx = torch.tensor(face_idx, dtype=torch.long, device=device)

        # 其余可学习参数
        self._features_dc = nn.Parameter(
            torch.tensor(f_dc, dtype=torch.float32, device=device).transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(f_rest, dtype=torch.float32, device=device).transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device=device).requires_grad_(True))

        # 全局 faces（不要从 ply 读），确保存在且为 long
        self._faces = torch.tensor(self.model.faces.astype(np.int64), dtype=torch.long, device=device)

        # 可选：加载 shape
        try:
            shape_elem = plydata.elements[1]
            shape_names = [p.name for p in shape_elem.properties if p.name.startswith("shape_")]
            shape_names = sorted(shape_names, key=lambda x: int(x.split('_')[-1]))
            if len(shape_names) > 0:
                shape = np.zeros((1, len(shape_names)), dtype=np.float32)
                for i, name in enumerate(shape_names):
                    shape[:, i] = np.asarray(shape_elem[name]).astype(np.float32)
                self._shape = nn.Parameter(torch.tensor(shape, dtype=torch.float32, device=device).requires_grad_(True))
        except Exception:
            pass

        self.active_sh_degree = self.max_sh_degree
        self.num_gs = self._uv_logits.shape[0]
        self.max_radii2D = torch.zeros((self.num_gs), device=device)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.num_gs, 1), device="cuda")
        self.denom = torch.zeros((self.num_gs, 1), device="cuda")

        scale = 1.0
        scale_small = 1.0
        l = [
            {'params': [self._uv_logits], 'lr': training_args.position_lr_init * self.spatial_lr_scale * scale_small,
             "name": "uv_logits"},
            {'params': [self._d], 'lr': training_args.position_lr_init * self.spatial_lr_scale * scale_small,
             "name": "d"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * scale, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr * scale / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr * scale_small, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * scale_small, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * scale_small, "name": "rotation"},
            {'params': [self._shape], 'lr': training_args.shape_lr * scale_small, "name": "flame_shape"}
        ]
        atlas_lr = getattr(training_args, "uv_atlas_lr", training_args.feature_lr)
        head_lr = getattr(training_args, "uv_head_lr", training_args.feature_lr)

        l.append({"params": [self.uv_feat], "lr": atlas_lr, "name": "uv_atlas"})
        l.append({"params": self.uv_smoother.parameters(), "lr": head_lr, "name": "uv_smoother"})
        l.append({"params": self.uv_heads.parameters(), "lr": head_lr, "name": "uv_heads"})

        freeze_point_app = getattr(training_args, "freeze_point_app", True)
        if freeze_point_app:
            for g in l:
                if g["name"] in ["f_dc", "f_rest", "opacity"]:
                    g["lr"] = 0.0

        self.params_list = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale * scale_small,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale * scale_small,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'flame_shape':
                continue
            # 跳过（uv_atlas / uv_smoother / uv_heads）
            if group["name"] not in tensors_dict:
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'flame_shape':
                continue
            # 跳过（uv_atlas / uv_smoother / uv_heads）
            if group["name"] not in ["uv_logits", "d", "f_dc", "f_rest", "opacity", "scaling", "rotation"]:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_uv_logits, new_d, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_faces_idx):
        d = {"uv_logits": new_uv_logits,
             "d": new_d,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._uv_logits = optimizable_tensors["uv_logits"]
        self._d = optimizable_tensors["d"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._face_idx = torch.cat([self._face_idx, new_faces_idx], dim=0)

        self.num_gs = self._uv_logits.shape[0]

        self.xyz_gradient_accum = torch.zeros((self.num_gs, 1), device="cuda")
        self.denom = torch.zeros((self.num_gs, 1), device="cuda")
        # self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=0)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.clamp_uv_logits_()
        if getattr(self, 'update_faces_on_densify', False):
            # Re-assign faces immediately so new points are rendered on the correct UV triangle.
            # This is important for TeGA-style 'free movement' across triangles.
            self.update_face_idx_from_uv()

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._uv_logits = optimizable_tensors["uv_logits"]
        self._d = optimizable_tensors["d"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.num_gs = self._uv_logits.shape[0]

        self._face_idx = self._face_idx[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.num_gs
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # topk_grad = torch.topk(grads, k=torch.tensor(self.num_gs * 0.1, dtype=torch.int), dim=0)[0]
        # selected_pts_mask = torch.where(grads >= topk_grad[-1], True, False)
        world_scale_max = self.get_world_scale_max_approx()
        selected_pts_mask = torch.logical_and(selected_pts_mask, world_scale_max > 0.01 * scene_extent)
        print(f"{selected_pts_mask.sum()} points are splitted")
        eps = 1e-6
        stds = self.scaling_activation(self._scaling[selected_pts_mask]).repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds / 100)  # 由于面片放缩的关系，太大会导致离群点
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        delta = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        # old uv in (0,1)
        uv_old = self._safe_sigmoid_uv(self._uv_logits[selected_pts_mask]).repeat(N, 1)  # (K*N,2)
        uv_new = (uv_old + delta[:, :2]).clamp(eps, 1.0 - eps)
        new_uv_logits = inverse_sigmoid(uv_new)
        new_d = self._d[selected_pts_mask].repeat(N, 1) + delta[:, 2:3]
        new_scaling = self.scaling_activation(self._scaling[selected_pts_mask].repeat(N, 1)) / (0.8 * N)
        new_scaling = self.scaling_inverse_activation(new_scaling)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_face_idx = self._face_idx[selected_pts_mask].repeat(N)
        # new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N) / (0.8 * N)

        # self.densification_postfix(new_uv_logits, new_d, new_features_dc, new_features_rest, new_opacity,
        #                            new_scaling, new_rotation, new_face_idx, new_max_radii2D)
        self.densification_postfix(new_uv_logits, new_d, new_features_dc, new_features_rest, new_opacity,
                                   new_scaling, new_rotation, new_face_idx)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # topk_grad = torch.topk(grads, k=torch.tensor(grads.shape[0] * 0.1, dtype=torch.int), dim=0)[0]
        # selected_pts_mask = torch.where(grads >= topk_grad[-1], True, False)
        # selected_pts_mask = torch.where(grads.squeeze() >= grad_threshold, True, False)
        # selected_pts_mask = torch.where(grads[0] >= grad_threshold, True, False)
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        world_scale_max = self.get_world_scale_max_approx()
        selected_pts_mask = torch.logical_and(selected_pts_mask, world_scale_max <= 0.01 * scene_extent)

        print(f"{selected_pts_mask.sum()} points are cloned")

        new_uv_logits = self._uv_logits[selected_pts_mask]
        new_d = self._d[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_face_idx = self._face_idx[selected_pts_mask]
        # new_max_radii2D = self.max_radii2D[selected_pts_mask]

        # self.densification_postfix(new_uv_logits, new_d, new_features_dc, new_features_rest, new_opacities, new_scaling,
        #                            new_rotation, new_face_idx, new_max_radii2D)
        self.densification_postfix(new_uv_logits, new_d, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_face_idx)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        denom = self.denom.clamp_min(1.0)
        grads = self.xyz_gradient_accum / denom
        grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        grads = torch.where(self.denom > 0.0, grads, torch.zeros_like(grads))

        max_total = getattr(self, "max_num_gaussians", 500_000)
        if self.num_gs >= max_total:
            return

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            world_scale_max = self.get_world_scale_max_approx()
            big_points_ws = world_scale_max > 0.03 * extent  # 0.05
            small_points_ws = world_scale_max < 0.00005
            # prune_mask = torch.logical_or(big_points_vs, big_points_ws)
            prune_mask = torch.logical_or(torch.logical_or(big_points_vs, big_points_ws), small_points_ws)
            print(f'{prune_mask.sum()} points are pruned')
            print(f'max_radii2D: {big_points_vs.sum()} | scaling: {big_points_ws.sum()}')
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_only(self, min_opacity=0.005, extent=0.01):
        unseen_points = (self.get_opacity < min_opacity).squeeze()
        world_scale_max = self.get_world_scale_max_approx()
        big_points_ws = world_scale_max > 0.03 * extent
        small_points_ws = world_scale_max < 0.00005
        prune_mask = torch.logical_or(torch.logical_or(unseen_points, big_points_ws), small_points_ws)
        self.prune_points(prune_mask)
        print(f'{prune_mask.sum()} points are pruned')
        print(f'opacity: {unseen_points.sum()} | big_scaling: {big_points_ws.sum()} | small_scaling: {small_points_ws.sum()}')

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        g = torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        g = F.normalize(g, dim=0)
        self.xyz_gradient_accum[update_filter] += g
        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def save_ckpt(self, path: str, optimizer=None, step: int = None, extra: dict = None):
        """保存：高斯逐点参数 + UV atlas网络参数 + 可选optimizer"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        ckpt = {
            "meta": {
                "step": int(step) if step is not None else None,
                "max_sh_degree": int(getattr(self, "max_sh_degree", 0)),
                "active_sh_degree": int(getattr(self, "active_sh_degree", getattr(self, "max_sh_degree", 0))),
                "jacobian_mode": getattr(self, "jacobian_mode", "autodiff"),
                "jacobian_create_graph": bool(getattr(self, "jacobian_create_graph", False)),
                "update_faces_on_densify": bool(getattr(self, "update_faces_on_densify", True)),
            },
            "gaussian_tensors": {},
            "extra_tensors": {},
            "net_state": {},
        }

        # ---- 逐点参数（这些就是 save_ply 里写出去的那一批）----
        # 你 save_ply 目前直接用 _opacity/_features_dc/... 写出 :contentReference[oaicite:1]{index=1}
        for name in [
            "_uv_logits", "_d", "_features_dc", "_features_rest",
            "_opacity", "_scaling", "_rotation",
            "_shape", "_expression", "_jaw_pose", "_leye_pose", "_reye_pose", "_neck_pose"
        ]:
            if hasattr(self, name):
                ckpt["gaussian_tensors"][name] = _to_cpu_tensor(getattr(self, name))

        # face index（非 Parameter，但很重要）
        if hasattr(self, "_face_idx"):
            ckpt["extra_tensors"]["_face_idx"] = _to_cpu_tensor(self._face_idx)

        # 一些状态张量（可选，不存也能重建，但存了断点更一致）
        for name in ["xyz_gradient_accum", "denom", "max_radii2D"]:
            if hasattr(self, name) and getattr(self, name) is not None:
                ckpt["extra_tensors"][name] = _to_cpu_tensor(getattr(self, name))

        # ---- UV atlas + smoother/head（如果你已经加了）----
        # uv_feat 是 Parameter，smoother/heads 是 nn.Module，各自存 state_dict
        if hasattr(self, "uv_feat"):
            ckpt["net_state"]["uv_feat"] = _to_cpu_tensor(self.uv_feat)

        if hasattr(self, "uv_smoother") and isinstance(self.uv_smoother, nn.Module):
            ckpt["net_state"]["uv_smoother"] = self.uv_smoother.state_dict()

        if hasattr(self, "uv_heads") and isinstance(self.uv_heads, nn.Module):
            ckpt["net_state"]["uv_heads"] = self.uv_heads.state_dict()

        # 一些 atlas 配置（恢复时要用）
        for k in ["uv_atlas_res", "uv_feat_dim", "uv_smoother_type"]:
            if hasattr(self, k):
                ckpt["meta"][k] = getattr(self, k)

        # ---- optimizer（可选）----
        if optimizer is not None:
            ckpt["optim_state"] = optimizer.state_dict()

        if extra is not None:
            ckpt["user_extra"] = extra

        torch.save(ckpt, path)

    def load_ckpt(self, path: str, optimizer=None, map_location="cpu", strict_net: bool = True):
        """加载：建议在 training_setup() 之前调用（否则 optimizer 里引用的 Parameter 会不匹配）"""
        ckpt = torch.load(path, map_location=map_location)
        meta = ckpt.get("meta", {})

        dev = getattr(self, "device", None)
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- 恢复逐点参数 ----
        g = ckpt.get("gaussian_tensors", {})
        def _restore_param(name, requires_grad=True):
            t = g.get(name, None)
            if t is None:
                return
            t = t.to(dev)
            if isinstance(getattr(self, name, None), nn.Parameter):
                setattr(self, name, nn.Parameter(t, requires_grad=requires_grad))
            else:
                # 某些是普通 tensor/变量，按原类型写回
                setattr(self, name, nn.Parameter(t, requires_grad=requires_grad))

        for name in ["_uv_logits", "_d", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
            _restore_param(name, requires_grad=True)

        # 可选：FLAME shape/pose 相关（如果你训练时在更新）
        for name in ["_shape", "_expression", "_jaw_pose", "_leye_pose", "_reye_pose", "_neck_pose"]:
            if name in g and g[name] is not None:
                _restore_param(name, requires_grad=True)

        # ---- face_idx / 状态张量 ----
        ex = ckpt.get("extra_tensors", {})
        if ex.get("_face_idx", None) is not None:
            self._face_idx = ex["_face_idx"].to(dev)

        # 这些状态不存也能重建；存了就恢复
        for name in ["xyz_gradient_accum", "denom", "max_radii2D"]:
            if ex.get(name, None) is not None:
                setattr(self, name, ex[name].to(dev))

        # 根据 uv_logits 数量更新 num_gs
        if hasattr(self, "_uv_logits") and self._uv_logits is not None:
            self.num_gs = int(self._uv_logits.shape[0])

        # ---- 恢复 UV atlas / smoother / heads ----
        ns = ckpt.get("net_state", {})
        if "uv_feat" in ns and hasattr(self, "uv_feat"):
            self.uv_feat = nn.Parameter(ns["uv_feat"].to(dev), requires_grad=True)

        if "uv_smoother" in ns and hasattr(self, "uv_smoother") and isinstance(self.uv_smoother, nn.Module):
            self.uv_smoother.load_state_dict(ns["uv_smoother"], strict=strict_net)

        if "uv_heads" in ns and hasattr(self, "uv_heads") and isinstance(self.uv_heads, nn.Module):
            self.uv_heads.load_state_dict(ns["uv_heads"], strict=strict_net)

        # ---- 恢复 meta ----
        if meta.get("active_sh_degree", None) is not None:
            self.active_sh_degree = int(meta["active_sh_degree"])

        # ---- optimizer（可选）----
        if optimizer is not None and "optim_state" in ckpt:
            optimizer.load_state_dict(ckpt["optim_state"])

        return meta