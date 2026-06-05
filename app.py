"""
Insurance Document Classifier — Streamlit App
Run: streamlit run app.py
"""

import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Make src importable
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import extract_text, SUPPORTED_EXTENSIONS
from src.feature_extractor import prepare_single_text, prepare_training_texts
from src.classifier import InsuranceClassifier, ALGORITHM_NAMES
from src.storage import (
    load_training_data,
    add_training_sample,
    delete_training_sample,
    clear_all_training_data,
    get_all_categories,
    get_subcategories_for,
)
from src.defaults import DEFAULT_CATEGORIES, DEFAULT_SUBCATEGORIES

MODEL_PATH = str(Path(__file__).parent / "models" / "classifier.joblib")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Document Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
def _load_classifier_from_disk() -> InsuranceClassifier | None:
    if os.path.exists(MODEL_PATH):
        try:
            return InsuranceClassifier.load(MODEL_PATH)
        except Exception:
            return None
    return None


if "classifier" not in st.session_state:
    st.session_state.classifier = _load_classifier_from_disk()

if "train_text" not in st.session_state:
    st.session_state.train_text = None
    st.session_state.train_filename = None

if "test_text" not in st.session_state:
    st.session_state.test_text = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Insurance\nDocument Classifier")
    st.divider()

    clf: InsuranceClassifier | None = st.session_state.classifier
    if clf and clf.is_trained:
        st.success(f"**Model:** {clf.classifier_name}")
        st.caption(f"Accuracy: {clf.training_accuracy:.1%} | Samples: {clf.n_samples_trained}")
    else:
        st.warning("No trained model yet")

    samples = load_training_data()
    st.info(f"Training samples: **{len(samples)}**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏋️ Train", "🔍 Test / Predict", "📊 Model Info"],
        label_visibility="collapsed",
    )

page_name = page.split(" ", 1)[1].strip()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRAIN
# ─────────────────────────────────────────────────────────────────────────────
if page_name == "Train":
    st.title("Training Data Management")
    st.caption("Upload labeled documents to build your training dataset, then train the classifier.")

    left, right = st.columns([1, 1], gap="large")

    # ── Left: Add sample ─────────────────────────────────────────────────────
    with left:
        st.subheader("Add Training Sample")

        ext_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        uploaded = st.file_uploader(
            f"Upload document ({ext_list})",
            type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
            key="train_uploader",
        )

        if uploaded:
            with st.spinner("Extracting text from document…"):
                try:
                    raw = uploaded.read()
                    text = extract_text(raw, uploaded.name)
                    st.session_state.train_text = text
                    st.session_state.train_filename = uploaded.name
                    st.success(f"Extracted {len(text):,} characters from **{uploaded.name}**")
                except Exception as exc:
                    st.error(f"Extraction failed: {exc}")
                    st.session_state.train_text = None

        if st.session_state.train_text:
            with st.expander("Preview extracted text", expanded=False):
                preview = st.session_state.train_text[:3000]
                if len(st.session_state.train_text) > 3000:
                    preview += "\n\n… (truncated)"
                st.text_area("", preview, height=160, disabled=True, label_visibility="collapsed")

            st.divider()

            # Category
            all_cats = get_all_categories()
            cat_options = all_cats + ["＋ Add new category…"]
            chosen_cat = st.selectbox("Category *", cat_options, key="sel_cat")

            if chosen_cat == "＋ Add new category…":
                category = st.text_input("New category name", key="new_cat").strip()
            else:
                category = chosen_cat

            # Subcategory (depends on category)
            if category and category != "＋ Add new category…":
                all_subcats = get_subcategories_for(category)
                subcat_options = all_subcats + ["＋ Add new subcategory…"]
                chosen_sub = st.selectbox("Subcategory *", subcat_options, key="sel_sub")
                if chosen_sub == "＋ Add new subcategory…":
                    subcategory = st.text_input("New subcategory name", key="new_sub").strip()
                else:
                    subcategory = chosen_sub
            else:
                subcategory = st.text_input("Subcategory *", key="sub_free").strip()

            # Keywords
            kw_raw = st.text_input(
                "Keywords (comma-separated, optional)",
                placeholder="e.g., premium, deductible, policy number",
                key="train_kw",
            )
            keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
            if keywords:
                st.caption(f"Keywords: {', '.join(keywords)}")

            can_add = bool(category and subcategory
                           and category != "＋ Add new category…"
                           and subcategory != "＋ Add new subcategory…")

            if st.button("➕ Add to Training Set", type="primary", disabled=not can_add):
                add_training_sample(
                    text=st.session_state.train_text,
                    filename=st.session_state.train_filename or "unknown",
                    category=category,
                    subcategory=subcategory,
                    keywords=keywords,
                )
                st.success(f"Added sample: **{category} / {subcategory}**")
                st.session_state.train_text = None
                st.session_state.train_filename = None
                st.rerun()

    # ── Right: Dataset & training ─────────────────────────────────────────────
    with right:
        st.subheader("Training Dataset")
        samples = load_training_data()

        if not samples:
            st.info("No training samples yet. Upload and label documents on the left.")
        else:
            # Summary metrics
            m1, m2, m3 = st.columns(3)
            label_counts = Counter(s["compound_label"] for s in samples)
            m1.metric("Total Samples", len(samples))
            m2.metric("Categories", len({s["category"] for s in samples}))
            m3.metric("Classes (cat/sub)", len(label_counts))

            # Table
            df_display = pd.DataFrame([
                {
                    "File": s["filename"],
                    "Category": s["category"],
                    "Subcategory": s["subcategory"],
                    "Keywords": ", ".join(s.get("keywords", [])) or "—",
                    "Date": s["timestamp"][:10],
                    "_id": s["id"],
                }
                for s in samples
            ])
            st.dataframe(
                df_display.drop(columns=["_id"]),
                use_container_width=True,
                height=220,
            )

            # Delete a sample
            with st.expander("Delete a sample"):
                del_options = {
                    f"{s['filename']} → {s['category']}/{s['subcategory']}": s["id"]
                    for s in samples
                }
                del_choice = st.selectbox("Select sample to remove", [""] + list(del_options.keys()))
                if del_choice and st.button("Delete", type="secondary"):
                    delete_training_sample(del_options[del_choice])
                    st.success("Sample deleted.")
                    st.rerun()

            with st.expander("⚠️ Clear ALL training data"):
                if st.button("Clear all training data", type="secondary"):
                    clear_all_training_data()
                    st.warning("All training data cleared.")
                    st.rerun()

            st.divider()

            # ── Train model ───────────────────────────────────────────────────
            st.subheader("Train Model")

            min_per_class = min(label_counts.values())
            if len(samples) < 2:
                st.warning("Add at least 2 training samples before training.")
            elif min_per_class < 1:
                st.warning("Each class needs at least 1 sample.")
            else:
                algo = st.selectbox("Algorithm", ALGORITHM_NAMES, key="algo_sel")

                # Class distribution chart
                dist_df = (
                    pd.DataFrame(list(label_counts.items()), columns=["Class", "Samples"])
                    .sort_values("Samples", ascending=False)
                )
                st.bar_chart(dist_df.set_index("Class"), height=180)

                if len(label_counts) == 1:
                    st.warning("All samples belong to one class. Add samples from more classes to train a meaningful classifier.")

                if st.button("🚀 Train Model", type="primary", disabled=len(label_counts) < 2):
                    with st.spinner(f"Training {algo}…"):
                        try:
                            texts = prepare_training_texts(samples)
                            labels = [s["compound_label"] for s in samples]

                            new_clf = InsuranceClassifier(classifier_name=algo)
                            new_clf.train(texts, labels)

                            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                            new_clf.save(MODEL_PATH)
                            st.session_state.classifier = new_clf

                            st.success(
                                f"Training complete!  "
                                f"Accuracy: **{new_clf.training_accuracy:.1%}**"
                            )
                            if new_clf.cv_scores is not None:
                                st.info(
                                    f"Cross-validation ({len(new_clf.cv_scores)}-fold): "
                                    f"{new_clf.cv_scores.mean():.1%} ± {new_clf.cv_scores.std():.1%}"
                                )
                        except Exception as exc:
                            st.error(f"Training failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TEST / PREDICT
# ─────────────────────────────────────────────────────────────────────────────
elif page_name == "Test / Predict":
    st.title("Test / Predict")
    st.caption("Upload a document and the trained model will classify it.")

    clf = st.session_state.classifier
    if not clf or not clf.is_trained:
        st.warning("⚠️ No trained model found. Go to **Train** and train a model first.")
        st.stop()

    uploaded = st.file_uploader(
        "Upload document to classify",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        key="test_uploader",
    )

    if uploaded:
        with st.spinner("Extracting text…"):
            try:
                raw = uploaded.read()
                text = extract_text(raw, uploaded.name)
                st.session_state.test_text = text
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                st.session_state.test_text = None

    if st.session_state.test_text:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader("Extracted Text")
            preview = st.session_state.test_text[:4000]
            if len(st.session_state.test_text) > 4000:
                preview += "\n\n… (truncated)"
            st.text_area("", preview, height=420, disabled=True, label_visibility="collapsed")

        with right:
            st.subheader("Classification")

            kw_raw = st.text_input(
                "Additional keywords to assist classification (optional)",
                placeholder="e.g., fire damage, claim number",
                key="test_kw",
            )
            test_keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]

            if st.button("🔍 Classify Document", type="primary"):
                with st.spinner("Classifying…"):
                    try:
                        prepared = prepare_single_text(st.session_state.test_text, test_keywords)
                        top = clf.predict_top_n(prepared, n=5)
                        best = top[0]

                        st.success("Classification complete!")
                        st.divider()

                        col_a, col_b = st.columns(2)
                        col_a.metric("Category", best["category"])
                        col_b.metric("Subcategory", best["subcategory"])

                        if best["confidence"] is not None:
                            conf = best["confidence"]
                            colour = "green" if conf >= 0.7 else ("orange" if conf >= 0.4 else "red")
                            st.progress(conf)
                            st.markdown(
                                f"**Confidence:** :{colour}[{conf:.1%}]"
                            )

                        if len(top) > 1:
                            st.divider()
                            st.subheader("All Predictions")
                            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                            rows = []
                            for i, pred in enumerate(top):
                                rows.append({
                                    "Rank": medals[i] if i < len(medals) else str(i + 1),
                                    "Category": pred["category"],
                                    "Subcategory": pred["subcategory"],
                                    "Confidence": f"{pred['confidence']:.1%}" if pred["confidence"] is not None else "—",
                                })
                            st.table(pd.DataFrame(rows).set_index("Rank"))
                    except Exception as exc:
                        st.error(f"Classification error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
elif page_name == "Model Info":
    st.title("Model Information")

    clf = st.session_state.classifier
    samples = load_training_data()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Samples", len(samples))
    col2.metric("Classes", len({s["compound_label"] for s in samples}) if samples else 0)
    if clf and clf.is_trained:
        col3.metric("Algorithm", clf.classifier_name)
        col4.metric(
            "Accuracy",
            f"{clf.training_accuracy:.1%}" if clf.training_accuracy is not None else "—",
        )
    else:
        col3.metric("Algorithm", "—")
        col4.metric("Accuracy", "—")

    if clf and clf.is_trained and clf.cv_scores is not None:
        st.info(
            f"Cross-validation ({len(clf.cv_scores)}-fold): "
            f"{clf.cv_scores.mean():.1%} ± {clf.cv_scores.std():.1%}  |  "
            f"Scores: {' / '.join(f'{s:.1%}' for s in clf.cv_scores)}"
        )

    if not samples:
        st.info("No training data yet.")
        st.stop()

    st.divider()

    # ── Class distribution ────────────────────────────────────────────────────
    st.subheader("Class Distribution")
    label_counts = Counter(s["compound_label"] for s in samples)
    dist_df = (
        pd.DataFrame(list(label_counts.items()), columns=["Class", "Count"])
        .sort_values("Count", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, max(3, len(label_counts) * 0.45)))
    bars = ax.barh(dist_df["Class"], dist_df["Count"], color="steelblue")
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Number of samples")
    ax.set_title("Samples per Class")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Confusion matrix on training data ─────────────────────────────────────
    if clf and clf.is_trained and len(label_counts) >= 2:
        st.subheader("Confusion Matrix (training data)")
        st.caption("Shows how the model predicts its own training data (optimistic — not a held-out evaluation).")
        with st.spinner("Computing…"):
            from sklearn.metrics import confusion_matrix
            texts = prepare_training_texts(samples)
            true_labels = [s["compound_label"] for s in samples]
            pred_labels = [clf.predict(t)["compound_label"] for t in texts]

            classes = sorted(set(true_labels))
            cm = confusion_matrix(true_labels, pred_labels, labels=classes)

            size = max(6, len(classes))
            fig2, ax2 = plt.subplots(figsize=(size, size))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                ax=ax2,
            )
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            ax2.set_title("Confusion Matrix")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    st.divider()

    # ── Full training data table ──────────────────────────────────────────────
    st.subheader("All Training Samples")
    df = pd.DataFrame([
        {
            "File": s["filename"],
            "Category": s["category"],
            "Subcategory": s["subcategory"],
            "Keywords": ", ".join(s.get("keywords", [])) or "—",
            "Date": s["timestamp"][:10],
        }
        for s in samples
    ])
    st.dataframe(df, use_container_width=True)

    # Download training data as CSV
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download training data (CSV)",
        data=csv,
        file_name="insurance_training_data.csv",
        mime="text/csv",
    )
