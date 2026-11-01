import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import conure_features as cf

# =========================================================
# Config
# =========================================================
MODEL_PATH = "model/conure_best.joblib"   # สร้างโดย compare_models.py
LOW_CONFIDENCE = 60                        # ต่ำกว่า % นี้จะขึ้นคำเตือน
MIN_MARGIN = 15                            # ผลอันดับ 1 นำอันดับ 2 น้อยกว่า % นี้ = กำกวม

st.set_page_config(page_title="Green-cheek Conure Color Classifier",
                   page_icon="🦜", layout="centered")
st.header("🦜 Green-cheek Conure Color Classification")
st.caption("อัปโหลดรูปนกแก้วกรีนชีค แล้วให้โมเดลทำนายสี (color mutation)")
st.divider()


# =========================================================
# Load Model
# =========================================================
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_everything():
    bundle = joblib.load(MODEL_PATH)
    bb = cf.load_backbone(bundle["backbone"]) if bundle["backbone"] else None
    gate_model = cf.load_imagenet_gate() if bundle.get("gate") else None
    return bundle, bb, gate_model


try:
    bundle, bb, gate_model = load_everything()
except Exception as e:
    st.error("❌ ไม่สามารถโหลด Model ได้")
    st.exception(e)
    st.info("ตรวจว่ามีไฟล์ `model/conure_best.joblib` (ได้จากการรัน compare_models.py) "
            "และติดตั้ง scikit-learn เวอร์ชันเดียวกับตอนเทรน")
    st.code(MODEL_PATH)
    st.stop()

pipeline = bundle["pipeline"]
CLASS_NAMES = bundle["class_names"]
cv = bundle.get("cv", {})


def extract(img):
    """คืน (embedding ล้วนสำหรับ gate, ฟีเจอร์สำหรับ classifier)"""
    if bb is None:                       # โมเดลแบบใช้สีอย่างเดียว
        return None, cf.color_features_batch([img])
    return cf.featurize_with_cnn([img], bb, bundle["use_color"])


def check_is_conure(img, cnn):
    """ตัวกรอง: คืน (ผ่านไหม, เหตุผลที่ไม่ผ่าน, ค่าวินิจฉัย)"""
    g = bundle.get("gate")
    if not g:
        return True, [], {}
    reasons, info = [], {}
    bird = float(cf.bird_scores([img], gate_model)[0])
    info["bird_score"], info["t_bird"] = bird, g["t_bird"]
    if bird < g["t_bird"]:
        reasons.append("ไม่พบลักษณะของนกในภาพ (อาจเป็นสัตว์อื่นหรือสิ่งของ)")
    if g.get("ref") is not None and cnn is not None and g.get("t_sim") is not None:
        sim = float(cf.knn_similarity(cf.l2norm(cnn), g["ref"].astype(np.float32), k=g["k"])[0])
        info["similarity"], info["t_sim"] = sim, g["t_sim"]
        if sim < g["t_sim"]:
            reasons.append("ภาพไม่คล้ายนกแก้วกรีนชีคที่โมเดลเคยเรียนรู้ (อาจเป็นนกหรือนกแก้วชนิดอื่น)")
    return not reasons, reasons, info


# =========================================================
# Upload & Predict
# =========================================================
st.subheader("📤 Upload an Image")
uploaded = st.file_uploader("Choose a green-cheek conure image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    raw = Image.open(uploaded)
    image = cf.prepare(raw)
    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption="Input Image", width="stretch")
    with c2:
        st.info(f"**File:** {uploaded.name}\n\n**Size:** {raw.size[0]} × {raw.size[1]} pixels")
    st.divider()

    if not bundle.get("gate"):
        st.warning("⚠️ ยังไม่ได้ตั้งค่าตัวกรองภาพที่ไม่ใช่กรีนชีค (รัน calibrate_gate.py) "
                   "ภาพนกชนิดอื่นหรือสัตว์อื่นจึงอาจได้ผลทำนายสีมั่วๆ")

    if st.button("🔍 Predict", type="primary", width="stretch"):
        with st.spinner("Model is analyzing the image..."):
            cnn, X = extract(image)
            is_conure, reasons, info = check_is_conure(image, cnn)
            probs = pipeline.predict_proba(X)[0] if is_conure else None

        if not is_conure:
            st.error("🚫 ภาพนี้ดูไม่เหมือนนกแก้วกรีนชีค จึงไม่แสดงผลทำนายสี")
            for r in reasons:
                st.write(f"- {r}")
            st.caption("ลองอัปโหลดภาพที่เห็นตัวนกกรีนชีคชัดเจน ไม่ถูกบัง และไม่ไกลเกินไป")
            with st.expander("🔧 ค่าวินิจฉัยของตัวกรอง"):
                st.json(info)
            st.stop()

        order = np.argsort(probs)[::-1]
        top, second = int(order[0]), int(order[1])
        conf = float(probs[top]) * 100
        margin = (float(probs[top]) - float(probs[second])) * 100

        st.subheader("🎯 Prediction Result")
        st.success(f"🦜 The model predicts: {CLASS_NAMES[top].upper()}")
        if conf < LOW_CONFIDENCE or margin < MIN_MARGIN:
            st.warning(
                f"⚠️ โมเดลยังไม่แน่ใจ (อันดับ 2 คือ {CLASS_NAMES[second]} ห่างกัน {margin:.0f}%) "
                "ลองใช้ภาพที่แสงสม่ำเสมอ เห็นตัวนกชัด ไม่มีฟิลเตอร์สี หรือถ่ายหลายมุมแล้วเทียบกัน"
            )
        m1, m2 = st.columns(2)
        m1.metric("Prediction", CLASS_NAMES[top])
        m2.metric("Confidence", f"{conf:.1f}%")
        st.progress(int(conf))

        st.write("**Top 3 predictions**")
        for i in order[:3]:
            st.write(f"- {CLASS_NAMES[i]}: {probs[i] * 100:.1f}%")
        with st.expander("📊 Show Prediction Details"):
            df = pd.DataFrame({"Class": CLASS_NAMES, "Probability (%)": probs * 100}).set_index("Class")
            st.bar_chart(df)
else:
    st.info("👆 Please upload an image of a green-cheek conure to begin.")

# =========================================================
# About
# =========================================================
st.divider()
with st.expander("ℹ️ About this application"):
    est = ""
    if cv:
        est = (f"ความแม่นยำโดยประมาณจาก cross-validation: **{cv['acc_source']:.0%} ± {cv['ci95']:.0%}** "
               f"(n={cv['n_sources']} ภาพต้นฉบับ, ฟีเจอร์: {cv['features']})")
    st.write(
        f"""
        **Classes:** {", ".join(CLASS_NAMES)}

        {est}

        **Note:** ผลทำนายอิงจากสีที่เห็นในภาพเท่านั้น แสง/กล้อง/ฟิลเตอร์ทำให้สีเพี้ยนได้
        และไม่สามารถบอกยีนแฝง (split) ได้ ควรใช้เป็นตัวช่วยประกอบการตัดสินใจ
        """
    )
