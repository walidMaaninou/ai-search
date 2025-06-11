import streamlit as st
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer, models
import numpy as np
import re
from collections import defaultdict
import statistics

st.markdown(
    """
    <style>
    /* جعل الصفحة كاملة اتجاهها من اليمين لليسار */
    html, body, [class*="css"]  {
        direction: rtl !important;
        unicode-bidi: embed;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    model_name = "aubmindlab/bert-base-arabertv2"
    
    # Load transformer model
    word_embedding_model = models.Transformer(model_name)

    # Use CLS token pooling
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

model = load_model()

def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)
    return text

def get_embedding(text):
    if not text:
        return [0.0] * 768
    text = normalize_arabic(text)
    embedding = model.encode(text)
    return embedding.tolist()

client = OpenSearch(
    hosts=[{'host': 'search-stg-lexisearch-opensearch-4kxlcycqij5qd63hx37xqazqi4.aos.us-east-1.on.aws', 'port': 443}],
    http_auth=('api-user', 'R@NkEDcT6.C_cCKqKWTu'),
    use_ssl=True,
    verify_certs=True,
    timeout=30
)

INDEX_NAME = "lexical_resources"

def fetch_parents_by_ids(parent_ids):
    try:
        docs = client.mget(body={"ids": parent_ids}, index=INDEX_NAME)
        parents_map = {}
        for doc in docs["docs"]:
            if doc["found"]:
                source = doc["_source"]
                entries = source.get("entries", [])
                headword = entries[0].get("headword", "غير متوفر") if entries else "غير متوفر"
                embedding = source.get("embedding_headword", None)
                parents_map[doc["_id"]] = {
                    "headword": headword,
                    "embedding_headword": embedding
                }
            else:
                parents_map[doc["_id"]] = {
                    "headword": "غير متوفر",
                    "embedding_headword": None
                }
        return parents_map
    except Exception as e:
        st.error(f"خطأ أثناء جلب البيانات: {e}")
        return {pid: {"headword": "غير متوفر", "embedding_headword": None} for pid in parent_ids}

def vector_search_with_definitions_only(query, fetch_k=3000):
    query_vec = get_embedding(query)

    query_body = {
        "size": fetch_k,
        "query": {
            "bool": {
                "filter": [
                    {"exists": {"field": "embedding_definition"}}
                ],
                "should": [
                    {
                        "knn": {
                            "embedding_headword": {
                                "vector": query_vec,
                                "k": fetch_k
                            }
                        }
                    },
                    {
                        "knn": {
                            "embedding_definition": {
                                "vector": query_vec,
                                "k": fetch_k
                            }
                        }
                    }
                ]
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=query_body)
    hits = response["hits"]["hits"]

    ranked = []
    q = np.array(query_vec)

    for hit in hits:
        source = hit["_source"]
        h_vec = source.get("embedding_headword")
        d_vec = source.get("embedding_definition")

        if not h_vec or not d_vec:
            continue

        sim_h = np.dot(h_vec, q) / (np.linalg.norm(h_vec) * np.linalg.norm(q))
        sim_d = np.dot(d_vec, q) / (np.linalg.norm(d_vec) * np.linalg.norm(q))
        avg_sim = (sim_h + sim_d) / 2.0

        hit["_score_headword"] = sim_h
        hit["_score_definition"] = sim_d
        hit["_custom_score"] = avg_sim
        ranked.append(hit)

    return sorted(ranked, key=lambda x: x["_custom_score"], reverse=True)


st.title("الكشاف للبحث بالذكاء الاصطناعي")

query = st.text_input("أدخل كلمة البحث:")
if st.button("بحث") and query:
    with st.spinner("جاري البحث..."):
        results = vector_search_with_definitions_only(query)

    if not results:
        st.warning("لم يتم العثور على نتائج.")
    else:
        grouped_results = defaultdict(list)

        for hit in results:
            try:
                parent_id = [
                    relation["ref"].split("_")[0]
                    for relation in hit["_source"].get("relations", [])[0].get("members", [])
                    if relation["role"] == "whole"
                ][0]
            except Exception:
                parent_id = "غير معروف"

            grouped_results[parent_id].append(hit)

        parent_ids = list(grouped_results.keys())
        parents_map = fetch_parents_by_ids(parent_ids)
        q = np.array(get_embedding(query))

        parent_scores = []

        for parent_id, hits in grouped_results.items():
            if not hits:
                continue
            avg_child_score = statistics.mean(hit.get("_custom_score", 0) for hit in hits)

            parent_info = parents_map.get(parent_id, {"embedding_headword": None})
            parent_embedding = parent_info.get("embedding_headword")

            if parent_embedding is not None:
                try:
                    parent_sim = np.dot(parent_embedding, q) / (np.linalg.norm(parent_embedding) * np.linalg.norm(q))
                except Exception:
                    parent_sim = 0
            else:
                parent_sim = 0

            PARENT_WEIGHT = 0.8
            CHILD_WEIGHT = 1.0 - PARENT_WEIGHT

            final_avg_score = (CHILD_WEIGHT * avg_child_score) + (PARENT_WEIGHT * parent_sim)


            parent_scores.append((parent_id, hits, avg_child_score, parent_sim, final_avg_score))

        sorted_groups = sorted(parent_scores, key=lambda x: x[4], reverse=True)

        st.success("النتائج مجمعة حسب الأب (مرتبة حسب المتوسط المشترك):")

        counter = 1
        for parent_id, hits, avg_child_score, parent_sim, final_avg_score in sorted_groups:
            parent_info = parents_map.get(parent_id, {"headword": "غير متوفر"})
            parent_headword = parent_info.get("headword", "غير متوفر")

            st.markdown(f"## {counter} — {parent_headword} — النتيجة: {final_avg_score:.4f}")
            st.markdown(f"- متوسط نتائج الأبناء: {avg_child_score:.4f}")
            st.markdown(f"- نتيجة الأب: {parent_sim:.4f}")

            counter += 1
            for i, hit in enumerate(hits, start=1):
                entries = hit["_source"].get("entries", [])
                headword = entries[0].get("headword", "غير متوفر") if entries else "غير متوفر"

                score_h = hit.get("_score_headword", 0)
                score_d = hit.get("_score_definition", 0)

                definition = "—"
                if entries:
                    senses = entries[0].get("senses", [])
                    if senses and senses[0].get("definitions"):
                        definition = senses[0]["definitions"][0].get("text", "—")

                st.markdown(f"### {i}. {headword}")
                st.markdown(f"- **درجة الكلمة:** {score_h:.4f}")
                st.markdown(f"- **درجة التعريف:** {score_d:.4f}")
                # إذا أردت عرض التعريف يمكنك إلغاء التعليق التالي
                st.markdown(f"- **التعريف:** {definition}")
                st.markdown("---")
