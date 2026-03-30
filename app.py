import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import random
import csv
import os
import requests
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import ast
import json

# ──────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Étude sur les recommandations de films",
    page_icon="🎬",
    layout="centered"
)

# ──────────────────────────────────────────────
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")

    def extract_names(json_str, key="name", limit=5):
        try:
            items = ast.literal_eval(json_str)
            return " ".join([item[key] for item in items[:limit]])
        except:
            return ""

    df["genres_clean"]   = df["genres"].apply(lambda x: extract_names(x))
    df["keywords_clean"] = df["keywords"].apply(lambda x: extract_names(x))
    df["overview"]       = df["overview"].fillna("")
    df["tagline"]        = df["tagline"].fillna("") if "tagline" in df.columns else ""
    df["vote_average"]   = df["vote_average"].fillna(0)
    df["popularity"]     = df["popularity"].fillna(0) if "popularity" in df.columns else 0
    df["features"]       = df["genres_clean"] + " " + df["keywords_clean"]
    df = df[df["genres_clean"] != ""].reset_index(drop=True)
    return df


@st.cache_data
def build_similarity_matrix(df):
    tfidf        = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["features"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


@st.cache_data
def get_tmdb_info(movie_id):
    """Récupère affiche + titre français en un seul appel TMDB."""
    try:
        api_key = st.secrets.get("TMDB_API_KEY", "")
        if not api_key:
            return None, None
        url      = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=fr-FR"
        response = requests.get(url, timeout=3)
        data     = response.json()
        poster   = f"https://image.tmdb.org/t/p/w300{data['poster_path']}" if data.get("poster_path") else None
        title_fr = data.get("title") or data.get("original_title") or None
        return poster, title_fr
    except:
        return None, None


@st.cache_data
def get_french_titles_bulk(movie_ids_tuple):
    """Récupère les titres français pour une liste de films (dict {movie_id: titre_fr})."""
    api_key = st.secrets.get("TMDB_API_KEY", "")
    if not api_key:
        return {}
    result = {}
    for movie_id in movie_ids_tuple:
        try:
            url      = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=fr-FR"
            response = requests.get(url, timeout=3)
            data     = response.json()
            title_fr = data.get("title") or data.get("original_title")
            if title_fr:
                result[movie_id] = title_fr
        except:
            pass
    return result


# ──────────────────────────────────────────────
# 2. ALGORITHME DE RECOMMANDATION
# ──────────────────────────────────────────────
def get_recommendations(liked_films, df, cosine_sim, n=6):
    indices     = pd.Series(df.index, index=df["title"]).drop_duplicates()
    sim_scores  = np.zeros(len(df))
    found_count = 0

    for film in liked_films:
        if film in indices:
            sim_scores  += cosine_sim[indices[film]]
            found_count += 1

    if found_count == 0:
        return []

    sim_scores    = sim_scores / found_count
    liked_indices = [indices[f] for f in liked_films if f in indices]
    sim_series    = pd.Series(sim_scores).drop(liked_indices, errors="ignore")
    top_indices   = sim_series.nlargest(n).index.tolist()

    user_genres     = set()
    user_keywords   = set()
    liked_overviews = []
    for lf in liked_films:
        if lf in indices:
            d = df.iloc[indices[lf]]
            user_genres.update(d["genres_clean"].split())
            user_keywords.update(d["keywords_clean"].split())
            if d["overview"]:
                liked_overviews.append(f"{lf}: {d['overview'][:120]}")

    recommendations = []
    for idx in top_indices:
        film            = df.iloc[idx]
        film_genres     = set(film["genres_clean"].split())
        film_keywords   = set(film["keywords_clean"].split())
        common_genres   = user_genres   & film_genres
        common_keywords = user_keywords & film_keywords

        reasons = []
        if common_genres:
            reasons.append(f"genres communs : {', '.join(list(common_genres)[:3])}")
        if common_keywords:
            reasons.append(f"thèmes similaires : {', '.join(list(common_keywords)[:3])}")
        if not reasons:
            reasons.append(f"style similaire (genres : {film['genres_clean']})")

        movie_id = int(film["id"]) if "id" in df.columns else None

        recommendations.append({
            "title":           film["title"],
            "genres":          film["genres_clean"],
            "keywords":        film["keywords_clean"],
            "overview":        film["overview"],
            "vote_average":    film["vote_average"],
            "tagline":         film.get("tagline", ""),
            "common_genres":   ", ".join(list(common_genres)[:4]),
            "common_keywords": ", ".join(list(common_keywords)[:5]),
            "liked_overviews": liked_overviews,
            "reasons":         reasons,
            "correct_reason":  reasons[0],
            "movie_id":        movie_id,
        })

    return recommendations


# ──────────────────────────────────────────────
# 3. GÉNÉRATION EXPLICATION + QCM PAR L'IA
# ──────────────────────────────────────────────
def generate_explanation_and_qcm(film, liked_films):
    """
    Claude génère en une fois :
    - Une explication courte et naturelle (longueur variable, 1-2 phrases)
    - La bonne réponse QCM spécifique au film
    - Deux distracteurs plausibles mais faux
    """
    client     = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    liked_text = ", ".join(liked_films[:5])
    overviews  = "\n".join(film["liked_overviews"][:3])

    context = []
    if film["common_genres"]:
        context.append(f"Genres communs : {film['common_genres']}")
    if film["common_keywords"]:
        context.append(f"Thèmes communs : {film['common_keywords']}")
    if film["overview"]:
        context.append(f"Synopsis : {film['overview'][:250]}")
    if film["tagline"]:
        context.append(f"Tagline : {film['tagline']}")

    prompt = f"""Tu es un assistant de recommandation de films.

Films aimés par l'utilisateur : {liked_text}
Synopses des films aimés : {overviews}

Film recommandé : {film['title']}
{chr(10).join(context)}

Génère UNIQUEMENT ce JSON valide (sans markdown, sans texte autour) :
{{
  "explanation": "1 à 2 phrases maximum en français, à la deuxième personne (vous), commençant par 'Ce film vous est recommandé car...'. Identifie la vraie connexion thématique ou narrative. Sois naturel et concis, varie la longueur.",
  "correct": "Phrase courte (max 12 mots) résumant la vraie raison. Ex: 'Pour ses thèmes de...' ou 'En raison de sa...'",
  "wrong1": "Distacteur plausible mais faux et spécifique à ce film. Max 12 mots.",
  "wrong2": "Autre distacteur crédible et différent. Max 12 mots."
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=350,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return json.loads(message.content[0].text)
    except:
        return {
            "explanation": message.content[0].text[:300],
            "correct":     film["correct_reason"],
            "wrong1":      "Parce qu'il est très populaire en ce moment",
            "wrong2":      "Parce qu'il correspond à votre tranche d'âge"
        }


# ──────────────────────────────────────────────
# 4. SAUVEGARDE DES RÉPONSES
# ──────────────────────────────────────────────
SHEET_COLUMNS = [
    "participant_id", "timestamp",
    "film_numero", "film_titre", "avec_explication", "texte_explication",
    "comprehension_percue_1", "comprehension_percue_2",
    "comprehension_reelle_bonne_reponse", "comprehension_reelle_reponse", "comprehension_reelle_correcte",
    "confiance_1", "confiance_2", "confiance_3",
    "films_aimes"
]

SHEET_URL = "https://docs.google.com/spreadsheets/d/1uX8KhfH4FcVKGXso8OG0nEEm8pn_GP6PjvltYbSF9UI/edit"

def get_sheet():
    """Connexion au Google Sheet via le compte de service."""
    scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds  = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_url(SHEET_URL).sheet1

def save_response(data):
    """Sauvegarde une ligne dans Google Sheets — permanent et en temps réel."""
    try:
        sheet = get_sheet()
        row   = [str(data.get(col, "")) for col in SHEET_COLUMNS]
        sheet.append_row(row, value_input_option="RAW")
    except Exception as e:
        st.warning(f"⚠️ Erreur de sauvegarde Google Sheets : {e}")


# ──────────────────────────────────────────────
# 5. INTERFACE STREAMLIT
# ──────────────────────────────────────────────
def main():
    defaults = {
        "step":              "welcome",
        "current_film_idx":  0,
        "recommendations":   [],
        "explanation_order": [],
        "liked_films":       [],
        "liked_films_temp":  [],
        "participant_id":    "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    try:
        df         = load_data()
        cosine_sim = build_similarity_matrix(df)
    except FileNotFoundError:
        st.error("❌ Fichier **tmdb_5000_movies.csv** introuvable. Placez-le dans le même dossier que app.py.")
        st.stop()

    # ── ACCUEIL ──────────────────────────────────
    if st.session_state.step == "welcome":
        st.title("🎬 Étude sur les systèmes de recommandation de films")
        st.markdown("""
Bienvenue et merci de participer à cette étude !

Cette recherche est conduite au sein du **Centre de Recherche en Informatique (CRI)**
de l'Université Paris 1 Panthéon-Sorbonne. Elle vise à mieux comprendre comment
les utilisateurs perçoivent et évaluent les recommandations générées par une IA dans un système de recommandation.

**Durée estimée : 6 à 8 minutes**

**Déroulement :**
1. Vous indiquez quelques films que vous avez aimés
2. Le système vous propose **10 recommandations** personnalisées
3. Après chaque recommandation, vous répondez à quelques questions courtes

🔑 **Vos réponses sont entièrement anonymes** et utilisées uniquement à des fins de recherche scientifique.
        """)
        st.session_state.participant_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if st.button("Commencer l'étude →", type="primary", use_container_width=True):
            st.session_state.step = "select_films"
            st.rerun()

    # ── SÉLECTION DES FILMS ───────────────────────
    elif st.session_state.step == "select_films":
        st.title("Vos films préférés")
        st.write("Recherchez et sélectionnez entre **3 et 5 films** que vous avez aimés :")
        st.caption("ℹ️ Les titres sont affichés en anglais (titre original international).")

        # Tous les films triés par popularité
        sorted_df  = df.sort_values("popularity", ascending=False).reset_index(drop=True)

        # Charger les titres français pour les 300 films les plus populaires
        if "french_titles" not in st.session_state:
            with st.spinner("Chargement des films..."):
                top_ids = tuple(int(i) for i in sorted_df["id"].head(300).tolist() if pd.notna(i))
                st.session_state.french_titles = get_french_titles_bulk(top_ids)

        fr_titles = st.session_state.french_titles

        # Construire la liste avec titres français quand disponibles
        # On garde un mapping titre_affiché -> titre_original pour l'algo
        display_to_original = {}
        display_titles = []
        for _, row in sorted_df.head(300).iterrows():
            movie_id   = int(row["id"]) if pd.notna(row.get("id")) else None
            title_fr   = fr_titles.get(movie_id, row["title"]) if movie_id else row["title"]
            display_to_original[title_fr] = row["title"]
            display_titles.append(title_fr)

        # Dédoublonner
        seen = set()
        unique_display = []
        for t in display_titles:
            if t not in seen:
                seen.add(t)
                unique_display.append(t)

        st.session_state.display_to_original = display_to_original

        selected_display = st.multiselect(
            "Tapez un titre pour rechercher parmi tous les films :",
            options=unique_display,
            max_selections=5,
            placeholder="Ex: Avatar, Titanic, Inception..."
        )
        # Reconvertir en titres originaux pour l'algorithme
        selected = [display_to_original.get(t, t) for t in selected_display]

        if len(selected) >= 3:
            st.success(f"✅ {len(selected)} film(s) sélectionné(s)")
            if st.button("Voir mes recommandations →", type="primary", use_container_width=True):
                with st.spinner("Calcul des recommandations..."):
                    recs = get_recommendations(selected, df, cosine_sim, n=10)
                if len(recs) < 6:
                    st.warning("Pas assez de recommandations. Essayez d'autres films.")
                else:
                    st.session_state.liked_films       = selected
                    st.session_state.recommendations   = recs
                    st.session_state.explanation_order = [True, True, True, True, True, False, False, False, False, False]
                    st.session_state.current_film_idx  = 0
                    st.session_state.step              = "recommendations"
                    st.rerun()
        elif len(selected) > 0:
            st.info(f"Sélectionnez encore {3 - len(selected)} film(s) pour continuer.")
        else:
            st.info("Commencez à taper un titre dans la barre de recherche.")

    # ── RECOMMANDATIONS + QUESTIONS ───────────────
    elif st.session_state.step == "recommendations":
        idx   = st.session_state.current_film_idx
        total = 10

        if idx >= total:
            st.session_state.step = "finished"
            st.rerun()
            return

        film             = st.session_state.recommendations[idx]
        show_explanation = st.session_state.explanation_order[idx]

        st.progress(idx / total)
        st.caption(f"Recommandation {idx + 1} sur {total}")
        st.divider()

        # ── Affichage du film avec affiche ──
        col_img, col_info = st.columns([1, 2])
        with col_img:
            if film["movie_id"]:
                poster_url, title_fr = get_tmdb_info(film["movie_id"])
            else:
                poster_url, title_fr = None, None
            display_title = title_fr if title_fr else film["title"]
            if poster_url:
                st.image(poster_url, width=150)
            else:
                st.markdown("### 🎬")
        with col_info:
            st.subheader(display_title)
            if film["genres"]:
                st.caption(f"🎭 {film['genres']}")
            if film["vote_average"] > 0:
                st.caption(f"⭐ {film['vote_average']}/10")

        # ── Génération explication + QCM (une seule fois) ──
        if f"qcm_correct_{idx}" not in st.session_state:
            label = "Génération de l'explication..." if show_explanation else "Chargement..."
            with st.spinner(label):
                try:
                    result = generate_explanation_and_qcm(film, st.session_state.liked_films)
                    st.session_state[f"expl_{idx}"]        = result["explanation"]
                    st.session_state[f"qcm_correct_{idx}"] = result["correct"]
                    st.session_state[f"qcm_wrong1_{idx}"]  = result["wrong1"]
                    st.session_state[f"qcm_wrong2_{idx}"]  = result["wrong2"]
                except Exception as e:
                    st.error(f"Erreur API : {e}")
                    st.stop()

        if show_explanation:
            st.markdown("#### 🤖 Pourquoi ce film vous est recommandé")
            st.info(st.session_state[f"expl_{idx}"])
            current_explanation = st.session_state[f"expl_{idx}"]
        else:
            current_explanation = ""

        st.divider()
        st.write("### Vos réponses")

        # ── Compréhension perçue ──
        st.write("**Compréhension perçue**")
        fmt = lambda x: {1: "1 — Pas du tout", 2: "2", 3: "3 — Moyennement", 4: "4", 5: "5 — Tout à fait"}[x]
        cp1 = st.select_slider("Je comprends pourquoi ce film m'a été recommandé.",
                               options=[1,2,3,4,5], value=3, key=f"cp1_{idx}", format_func=fmt)
        cp2 = st.select_slider("Je suis capable d'expliquer la logique de cette recommandation.",
                               options=[1,2,3,4,5], value=3, key=f"cp2_{idx}", format_func=fmt)

        # ── Compréhension réelle (QCM fixe) ──
        st.write("**Test de compréhension**")
        if f"qcm_options_{idx}" not in st.session_state:
            correct = st.session_state[f"qcm_correct_{idx}"]
            wrong1  = st.session_state[f"qcm_wrong1_{idx}"]
            wrong2  = st.session_state[f"qcm_wrong2_{idx}"]
            wrong3  = "Je ne sais pas / la raison ne m'est pas claire"
            pool    = [correct, wrong1, wrong2]
            random.shuffle(pool)
            pool.append(wrong3)
            labels  = ["A", "B", "C", "D"]
            st.session_state[f"qcm_options_{idx}"] = {labels[i]: pool[i] for i in range(4)}

        qcm             = st.session_state[f"qcm_options_{idx}"]
        correct         = st.session_state[f"qcm_correct_{idx}"]
        options_display = [f"{k}. {v}" for k, v in qcm.items()]
        correct_display = next(f"{k}. {v}" for k, v in qcm.items() if v == correct)

        cr_display = st.radio("Selon vous, pourquoi ce film vous a-t-il été recommandé ?",
                              options_display, key=f"cr_{idx}", index=None)
        real_correct = (cr_display == correct_display) if cr_display is not None else None

        # ── Confiance ──
        st.write("**Confiance**")
        t1 = st.select_slider("Je fais confiance à cette recommandation.",
                               options=[1,2,3,4,5], value=3, key=f"t1_{idx}", format_func=fmt)
        t2 = st.select_slider("Ce film me semble pertinent pour moi.",
                               options=[1,2,3,4,5], value=3, key=f"t2_{idx}", format_func=fmt)
        t3 = st.select_slider("Ce système de recommandation me semble fiable.",
                               options=[1,2,3,4,5], value=3, key=f"t3_{idx}", format_func=fmt)

        st.divider()

        if cr_display is None:
            st.warning("Veuillez répondre au test de compréhension avant de continuer.")
        else:
            label = "Suivant →" if idx < total - 1 else "Terminer l'étude ✓"
            if st.button(label, type="primary", use_container_width=True):
                save_response({
                    "participant_id":                st.session_state.participant_id,
                    "timestamp":                     datetime.now().isoformat(),
                    "film_numero":                   idx + 1,
                    "film_titre":                    film["title"],
                    "avec_explication":              show_explanation,
                    "texte_explication":             current_explanation,
                    "comprehension_percue_1":        cp1,
                    "comprehension_percue_2":        cp2,
                    "comprehension_reelle_bonne_reponse": st.session_state.get(f"qcm_correct_{idx}", ""),
                    "comprehension_reelle_option_b":     st.session_state.get(f"qcm_wrong1_{idx}", ""),
                    "comprehension_reelle_option_c":     st.session_state.get(f"qcm_wrong2_{idx}", ""),
                    "comprehension_reelle_reponse":      cr_display,
                    "comprehension_reelle_correcte":     real_correct,
                    "confiance_1":                       t1,
                    "confiance_2":                       t2,
                    "confiance_3":                       t3,
                    "score_confiance":                   round((t1 + t2 + t3) / 3, 2),
                    "films_aimes":                   " | ".join(st.session_state.liked_films)
                })
                st.session_state.current_film_idx += 1
                st.rerun()

    # ── FIN ──────────────────────────────────────
    elif st.session_state.step == "finished":
        st.balloons()
        st.success("🎉 Merci pour votre participation !")
        st.markdown("""
Vos réponses ont bien été enregistrées.


        """)
        if st.button("Nouvelle participation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()