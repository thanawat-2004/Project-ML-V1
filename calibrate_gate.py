# =========================================================
# calibrate_gate.py
# ตั้งค่า "ตัวกรองภาพที่ไม่ใช่นกแก้วกรีนชีค" ให้โมเดลใน model/conure_best.joblib
#
# ใช้ 2 ด่าน:
#   1) คะแนน "เป็นนก" จาก ImageNet classifier   (ตัดสัตว์อื่น/สิ่งของ/ภาพไม่มีนก)
#   2) ความคล้ายกับภาพกรีนชีคที่เคยเรียน         (ตัดนกชนิดอื่นที่ดูต่างจากกรีนชีคมาก)
#
# (ทางเลือกแต่แนะนำมาก) สร้างโฟลเดอร์ negatives/ ใส่ภาพที่ "ไม่ใช่กรีนชีค" 100-300 ภาพ
#   เช่น นกแก้ว/นกชนิดอื่น (sun conure, cockatiel, budgie, lovebird, Pyrrhura ชนิดอื่น),
#   สัตว์อื่น (แมว หมา), สิ่งของ, กรง/พื้นหลังที่ไม่มีนก
#   สคริปต์จะใช้ภาพเหล่านี้เลือก threshold และรายงานว่าตัดออกได้กี่ %
#   ถ้าไม่มี negatives จะใช้ค่า percentile ที่ 1% ของภาพกรีนชีคจริง (ไม่รู้ว่าตัดของอื่นได้เท่าไร)
#
# รัน (หลังรัน compare_models.py แล้ว):  python calibrate_gate.py
# =========================================================
import pathlib

import joblib
import numpy as np

import compare_models as cmp
import conure_features as cf

MODEL_PATH = cmp.MODEL_OUT
NEG_DIR = pathlib.Path("negatives")
K = 5
TARGET_ACCEPT = 0.95    # สัดส่วนภาพกรีนชีคจริงที่ต้องผ่านด่าน (ใช้เมื่อมี negatives)
DEFAULT_Q = 0.01        # เมื่อไม่มี negatives: ตัดทิ้งได้ไม่เกิน ~1% ต่อด่าน
QS = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]


def list_negatives():
    if not NEG_DIR.exists():
        return []
    return sorted(p for p in NEG_DIR.rglob("*")
                  if p.is_file() and p.suffix.lower() in cmp.EXTS)


def topk_mean(S, k):
    k = min(k, S.shape[1])
    return np.sort(S, axis=1)[:, -k:].mean(axis=1)


def choose(bird_pos, sim_pos, bird_neg, sim_neg):
    if bird_neg is None:
        return (float(np.quantile(bird_pos, DEFAULT_Q)),
                float(np.quantile(sim_pos, DEFAULT_Q)) if sim_pos is not None else None)
    sim_p = sim_pos if sim_pos is not None else np.zeros_like(bird_pos)
    sim_n = sim_neg if sim_neg is not None else np.zeros_like(bird_neg)
    tbs = np.quantile(bird_pos, QS)
    tss = np.quantile(sim_p, QS) if sim_pos is not None else [-np.inf]
    best = None
    for tb in tbs:
        for ts in tss:
            acc = np.mean((bird_pos >= tb) & (sim_p >= ts))
            if acc < TARGET_ACCEPT:
                continue
            rej = 1 - np.mean((bird_neg >= tb) & (sim_n >= ts))
            if best is None or (rej, acc) > best[0]:
                best = ((rej, acc), float(tb), float(ts))
    if best is None:
        print("หมายเหตุ: ไม่มี threshold ที่ผ่านเงื่อนไข -> ใช้ค่า percentile 1%")
        return choose(bird_pos, sim_pos, None, None)
    return best[1], (best[2] if sim_pos is not None else None)


def main():
    bundle = joblib.load(MODEL_PATH)
    class_names, paths, y, groups = cmp.collect_images()
    bname = bundle["backbone"]
    negs = list_negatives()
    print(f"ภาพกรีนชีค {len(paths)} ไฟล์ | negatives {len(negs)} ไฟล์ | backbone: {bname}")

    gate_model = cf.load_imagenet_gate()
    bird_pos = cmp.cached("gate_bird_pos", paths,
                          lambda: cf.bird_scores_from_paths(paths, gate_model))
    bird_neg = (cmp.cached("gate_bird_neg", negs,
                           lambda: cf.bird_scores_from_paths(negs, gate_model))
                if negs else None)

    ref = sim_pos = sim_neg = None
    if bname:
        bb_holder = {}

        def emb(ps):
            if "bb" not in bb_holder:
                bb_holder["bb"] = cf.load_backbone(bname)
            return cf.cnn_features_from_paths(ps, bb_holder["bb"])

        E_pos = cf.l2norm(cmp.cached(bname, paths, lambda: emb(paths)))
        codes = np.unique(groups, return_inverse=True)[1]
        S = E_pos @ E_pos.T
        S[codes[:, None] == codes[None, :]] = -np.inf     # ตัดสำเนา/ภาพจาก source เดียวกันออก
        sim_pos = topk_mean(S, K)
        ref = E_pos
        if negs:
            E_neg = cf.l2norm(cmp.cached(bname + "_neg", negs, lambda: emb(negs)))
            sim_neg = topk_mean(E_neg @ E_pos.T, K)

    t_bird, t_sim = choose(bird_pos, sim_pos, bird_neg, sim_neg)

    ok_pos = (bird_pos >= t_bird) & ((sim_pos >= t_sim) if t_sim is not None else True)
    print("\n================ ผลการตั้งค่า ================")
    print(f"threshold คะแนนนก (ImageNet) : {t_bird:.3f}")
    print(f"threshold ความคล้าย (kNN)    : {t_sim if t_sim is None else round(t_sim, 3)}")
    print(f"ภาพกรีนชีคจริงที่ผ่าน         : {ok_pos.mean():.1%}  (ถูกตัดทิ้ง {int((~ok_pos).sum())}/{len(ok_pos)})")
    if negs:
        ok_neg = (bird_neg >= t_bird) & ((sim_neg >= t_sim) if t_sim is not None else True)
        print(f"ภาพที่ไม่ใช่กรีนชีคที่ถูกตัด    : {1 - ok_neg.mean():.1%}  (หลุดผ่าน {int(ok_neg.sum())}/{len(ok_neg)})")
        print(f"   - ตัดโดยด่านนกอย่างเดียว    : {np.mean(bird_neg < t_bird):.1%}")
        if t_sim is not None:
            print(f"   - ตัดโดยด่านความคล้ายอย่างเดียว: {np.mean(sim_neg < t_sim):.1%}")
        leaked = np.where(ok_neg)[0]
        if len(leaked):
            print("\nภาพ negatives ที่ยังหลุดผ่าน (ตรวจดูว่าควรเพิ่มภาพแบบนี้ใน negatives):")
            for i in leaked[:15]:
                s = f"sim={sim_neg[i]:.2f}" if sim_neg is not None else ""
                print(f"   {negs[i].name}  bird={bird_neg[i]:.2f} {s}")
    else:
        print("ไม่มีโฟลเดอร์ negatives/ -> ไม่ทราบว่าตัดภาพที่ไม่ใช่กรีนชีคได้กี่ % "
              "(แนะนำให้เพิ่ม negatives แล้วรันใหม่)")

    bundle["gate"] = {
        "t_bird": t_bird,
        "t_sim": t_sim,
        "k": K,
        "ref": ref.astype(np.float16) if ref is not None else None,
        "n_negatives": len(negs),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nบันทึกตัวกรองลงใน {MODEL_PATH} แล้ว -> รันแอปได้เลย")


if __name__ == "__main__":
    main()
