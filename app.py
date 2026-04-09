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
import re

st.set_page_config(
    page_title="Étude sur les recommandations de films",
    page_icon="🎬",
    layout="centered"
)

# ──────────────────────────────────────────────
# 1. DONNÉES
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
def base_title(title):
    """
    Extrait le titre de base d'un film en supprimant :
    - les chiffres (2, 3, VII...)
    - les mots de suite (part, chapter, episode, begins, returns, rises, forever, beyond...)
    - la ponctuation
    Retourne une chaîne normalisée pour comparaison.
    """
    sequel_words = {
        "part", "chapter", "episode", "vol", "volume", "ii", "iii", "iv",
        "vi", "vii", "viii", "ix", "xi", "xii", "begins", "returns",
        "rises", "forever", "beyond", "resurrection", "apocalypse",
        "reckoning", "reloaded", "revolutions", "origins", "legacy",
        "aftermath", "endgame", "ultron", "infinity", "civil", "winter",
        "dark", "last", "first", "new", "next", "again"
    }
    # Supprimer ponctuation et chiffres
    t = re.sub(r'[^a-zA-Z\s]', ' ', title.lower())
    words = t.split()
    # Garder uniquement les mots significatifs (pas sequel_words, pas chiffres, longueur > 2)
    core = [w for w in words if w not in sequel_words and len(w) > 2]
    return " ".join(core)

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

    # Titres de base des films aimés pour détecter les suites/franchises
    liked_bases = [base_title(lf) for lf in liked_films]

    def is_same_franchise(candidate_title):
        cb = base_title(candidate_title)
        if not cb:
            return False
        for lb in liked_bases:
            if not lb:
                continue
            # Si le titre de base du candidat contient celui du film aimé ou vice versa
            if lb in cb or cb in lb:
                return True
        return False

    # Demander plus de candidats pour compenser les exclusions
    filtered_series = sim_series[
        ~sim_series.index.map(lambda i: is_same_franchise(df.iloc[i]["title"]))
    ]
    top_indices = filtered_series.nlargest(n).index.tolist()

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
# 3. GÉNÉRATION EXPLICATION + QCM
# ──────────────────────────────────────────────
def generate_explanation_and_qcm(film, liked_films):
    import time
    client     = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    liked_text = ", ".join(liked_films[:5])
    overviews  = "\n".join(film["liked_overviews"][:3])

    context = []
    if film["common_genres"]:
        context.append(f"Genres communs avec les films aimés : {film['common_genres']}")
    if film["common_keywords"]:
        context.append(f"Mots-clés/thèmes communs : {film['common_keywords']}")
    if film["overview"]:
        context.append(f"Synopsis du film recommandé : {film['overview'][:300]}")
    if film["tagline"]:
        context.append(f"Tagline : {film['tagline']}")
    if film["vote_average"]:
        context.append(f"Note : {film['vote_average']}/10")

    prompt = f"""Tu es un expert en analyse cinématographique et en recommandation personnalisée.

--- PROFIL DE L'UTILISATEUR ---
Films aimés : {liked_text}

Synopses des films aimés :
{overviews}

--- FILM RECOMMANDÉ ---
Titre : {film['title']}
{chr(10).join(context)}

--- TA TÂCHE ---
Génère une explication de recommandation personnalisée en 3 temps :

TEMPS 1 — Analyse du profil (ne pas afficher) :
Identifie ce qui caractérise vraiment les goûts de cet utilisateur :
thèmes récurrents, style narratif, atmosphère, émotions recherchées.

TEMPS 2 — Connexion avec le film (ne pas afficher) :
Trouve les ponts réels entre ce film et le profil : narratifs, thématiques, stylistiques.
Ne pas répéter les genres mécaniquement.

TEMPS 3 — Rédige l'explication finale :
- 3 à 4 phrases naturelles en français, à la deuxième personne (vous)
- Commence par : "Au vu de vos films, vous semblez apprécier [caractéristique précise]..."
- Puis explique le lien avec ce film spécifique
- Mentionne au moins un film aimé par son nom
- Ne liste jamais les genres bruts, ne raconte pas le synopsis

Génère UNIQUEMENT ce JSON valide (sans markdown, sans texte autour) :
{{
  "explanation": "Ton explication ici (3-4 phrases, commence par l'analyse du profil)",
  "correct": "Phrase courte (max 15 mots) résumant la vraie connexion. Ex: 'Pour son atmosphère [X] et ses thèmes de [Y], similaires à [film aimé]'",
  "wrong1": "Distracteur plausible mais faux, spécifique à ce film. Max 15 mots.",
  "wrong2": "Autre distracteur crédible et différent. Max 15 mots."
}}"""

    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
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
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return {
                    "explanation": "Ce film vous est recommandé en raison de ses similarités avec vos préférences.",
                    "correct":     film["correct_reason"],
                    "wrong1":      "Parce qu'il est très populaire en ce moment",
                    "wrong2":      "Parce qu'il correspond à votre tranche d'âge"
                }

# ──────────────────────────────────────────────
# 4. SAUVEGARDE
# ──────────────────────────────────────────────
SHEET_COLUMNS = [
    "participant_id", "timestamp",
    "film_numero", "film_titre", "avec_explication", "texte_explication",
    "comprehension_percue_1", "comprehension_percue_2",
    "comprehension_reelle_bonne_reponse", "comprehension_reelle_reponse", "comprehension_reelle_correcte",
    "confiance_1", "confiance_2", "confiance_3",
    "films_aimes"
]

PROFILE_COLUMNS = [
    "participant_id", "timestamp_profil",
    "age", "profession", "pays"
]

PRE_COLUMNS = [
    "participant_id", "timestamp_pre",
    "pre_utilise_reco", "pre_confiance_reco", "pre_comprend_reco", "pre_importance_explication"
]

SHEET_URL = "https://docs.google.com/spreadsheets/d/1uX8KhfH4FcVKGXso8OG0nEEm8pn_GP6PjvltYbSF9UI/edit"

def get_sheet(sheet_index=0):
    scopes = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds  = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    wb     = client.open_by_url(SHEET_URL)
    sheets = wb.worksheets()
    return sheets[sheet_index] if sheet_index < len(sheets) else sheets[0]

def save_response(data):
    try:
        sheet = get_sheet(0)
        row   = [str(data.get(col, "")) for col in SHEET_COLUMNS]
        sheet.append_row(row, value_input_option="RAW")
    except Exception as e:
        st.warning(f"⚠️ Erreur sauvegarde : {e}")

def save_profile(data):
    try:
        sheet = get_sheet(1)
        row   = [str(data.get(col, "")) for col in PROFILE_COLUMNS]
        sheet.append_row(row, value_input_option="RAW")
    except Exception as e:
        st.warning(f"⚠️ Erreur sauvegarde profil : {e}")

def save_pre(data):
    try:
        sheet = get_sheet(2)
        row   = [str(data.get(col, "")) for col in PRE_COLUMNS]
        sheet.append_row(row, value_input_option="RAW")
    except Exception as e:
        st.warning(f"⚠️ Erreur sauvegarde pré-questionnaire : {e}")


# ──────────────────────────────────────────────
# 5. INTERFACE
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
        st.error("❌ Fichier **tmdb_5000_movies.csv** introuvable.")
        st.stop()

    # ── ACCUEIL ──────────────────────────────────
    if st.session_state.step == "welcome":
        st.title("🎬 Étude sur les systèmes de recommandation de films")
        st.markdown("""
Bienvenue et merci de participer à cette étude !

Cette recherche est conduite au sein de l'Université Paris 1 Panthéon-Sorbonne.
Elle vise à mieux comprendre comment
les utilisateurs perçoivent et évaluent les recommandations générées par une IA.

**Durée estimée : 8 à 10 minutes**

**Déroulement :**
1. Quelques questions sur vos habitudes avec les systèmes de recommandation
2. Vous indiquez des films que vous avez aimés
3. Le système vous propose **6 recommandations** personnalisées
4. Quelques questions sur votre profil

🔑 **Vos réponses sont entièrement anonymes** et utilisées uniquement à des fins de recherche scientifique.
        """)
        st.session_state.participant_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if st.button("Commencer →", type="primary", use_container_width=True):
            st.session_state.step = "pre_questionnaire"
            st.rerun()

    # ── PRÉ-QUESTIONNAIRE ─────────────────────────
    elif st.session_state.step == "pre_questionnaire":
        st.title("❓Les systèmes de recommandation")
        st.markdown("""
Un **système de recommandation** est un algorithme qui analyse vos préférences pour vous suggérer
automatiquement des contenus susceptibles de vous plaire, c'est ce que font Netflix, Spotify ou
YouTube lorsqu'ils vous proposent des films, des musiques ou des vidéos "pour vous".

Avant de commencer l'expérience, quelques questions sur votre rapport à ces systèmes.
        """)
        st.divider()

        st.markdown("**Question 1 sur 4**")
        st.markdown("###### Utilisez-vous des plateformes intégrant des systèmes de recommandation ? (ex. Netflix, YouTube, Spotify, Amazon...)")
        pre1 = st.radio(
            "",
            [
                "Oui, quotidiennement",
                "Oui, plusieurs fois par semaine",
                "Occasionnellement",
                "Non, je n'en utilise pas"
            ],
            index=None, key="pre1"
        )

        st.markdown("**Question 2 sur 4**")
        st.markdown("###### De manière générale, dans quelle mesure faites-vous confiance aux recommandations proposées par ces systèmes ?")
        pre2 = st.select_slider(
            "",
            options=[1, 2, 3, 4, 5],
            value=3,
            key="pre2",
            format_func=lambda x: {
                1: "1 — Aucune confiance",
                2: "2 — Peu de confiance",
                3: "3 — Confiance modérée",
                4: "4 — Assez confiance",
                5: "5 — Totale confiance"
            }[x]
        )

        st.markdown("**Question 3 sur 4**")
        st.markdown("###### Pensez-vous comprendre le fonctionnement des algorithmes qui génèrent ces recommandations ?")
        pre3 = st.radio(
            "",
            [
                "Oui, je comprends bien leur fonctionnement",
                "J'en ai une idée générale, sans en maîtriser les détails",
                "Je n'ai qu'une vague idée de leur fonctionnement",
                "Non, je ne comprends pas comment ils fonctionnent"
            ],
            index=None, key="pre3"
        )

        st.markdown("**Question 4 sur 4**")
        st.markdown("###### Selon vous, est-il important qu'un système de recommandation justifie ses suggestions en expliquant pourquoi un contenu vous est proposé ?")
        pre4 = st.select_slider(
            "",
            options=[1, 2, 3, 4, 5],
            value=3,
            key="pre4",
            format_func=lambda x: {
                1: "1 — Pas du tout important",
                2: "2 — Peu important",
                3: "3 — Assez important",
                4: "4 — Important",
                5: "5 — Très important"
            }[x]
        )

        st.divider()

        if pre1 is None or pre3 is None:
            st.warning("Veuillez répondre à toutes les questions avant de continuer.")
        else:
            if st.button("Continuer →", type="primary", use_container_width=True):
                save_pre({
                    "participant_id":             st.session_state.participant_id,
                    "timestamp_pre":              datetime.now().isoformat(),
                    "pre_utilise_reco":           pre1,
                    "pre_confiance_reco":         pre2,
                    "pre_comprend_reco":          pre3,
                    "pre_importance_explication": pre4,
                })
                st.session_state.step = "select_films"
                st.rerun()

    # ── SÉLECTION DES FILMS ───────────────────────
    elif st.session_state.step == "select_films":
        st.title("Vos films préférés")
        st.markdown("""
Dans cette étude, vous allez choisir quelques films que vous avez aimés.
Un système de recommandation va ensuite vous proposer **6 films** susceptibles de vous plaire.

Pour chaque recommandation, vous aurez à répondre à quelques questions sur votre compréhension
et votre ressenti vis-à-vis de la recommandation.

⬆️ *Après chaque réponse, pensez à remonter en haut de page pour accéder à la recommandation suivante.*
        """)
        st.divider()
        st.write("Sélectionnez entre **3 et 5 films** que vous avez aimés :")

        sorted_df  = df.sort_values("popularity", ascending=False).reset_index(drop=True)

        if "french_titles" not in st.session_state:
            with st.spinner("Chargement des films..."):
                top_ids = tuple(int(i) for i in sorted_df["id"].head(300).tolist() if pd.notna(i))
                st.session_state.french_titles = get_french_titles_bulk(top_ids)

        fr_titles = st.session_state.french_titles
        display_to_original = {}
        display_titles = []
        for _, row in sorted_df.head(300).iterrows():
            movie_id   = int(row["id"]) if pd.notna(row.get("id")) else None
            title_fr   = fr_titles.get(movie_id, row["title"]) if movie_id else row["title"]
            display_to_original[title_fr] = row["title"]
            display_titles.append(title_fr)

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
        selected = [display_to_original.get(t, t) for t in selected_display]

        if len(selected) >= 3:
            st.success(f"✅ {len(selected)} film(s) sélectionné(s)")
            if st.button("Voir mes recommandations →", type="primary", use_container_width=True):
                with st.spinner("Calcul des recommandations..."):
                    recs = get_recommendations(selected, df, cosine_sim, n=6)
                if len(recs) < 4:
                    st.warning("Pas assez de recommandations. Essayez d'autres films.")
                else:
                    flags = [True, True, True, False, False, False]
                    random.shuffle(flags)
                    st.session_state.liked_films       = selected
                    st.session_state.recommendations   = recs
                    st.session_state.explanation_order = flags
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
        total = 6

        if idx >= total:
            st.session_state.step = "profile"
            st.rerun()
            return

        film             = st.session_state.recommendations[idx]
        show_explanation = st.session_state.explanation_order[idx]

        # ── Écran de chargement propre ──
        if f"qcm_correct_{idx}" not in st.session_state:
            st.progress((idx + 1) / total)
            st.caption(f"Recommandation {idx + 1} sur {total}")
            st.divider()
            st.markdown(
                """<div style="text-align:center;padding:40px 0;">
                <div style="font-size:48px;">⏳</div>
                <div style="font-size:20px;font-weight:bold;margin-top:12px;">
                Chargement de la recommandation suivante...</div>
                <div style="color:gray;margin-top:8px;">Merci de patienter quelques secondes</div>
                </div>""",
                unsafe_allow_html=True
            )
            try:
                result = generate_explanation_and_qcm(film, st.session_state.liked_films)
                st.session_state[f"expl_{idx}"]        = result["explanation"]
                st.session_state[f"qcm_correct_{idx}"] = result["correct"]
                st.session_state[f"qcm_wrong1_{idx}"]  = result["wrong1"]
                st.session_state[f"qcm_wrong2_{idx}"]  = result["wrong2"]
            except Exception as e:
                st.error(f"Erreur API : {e}")
                st.stop()
            st.rerun()
            return

        # ── Affichage normal ──
        st.progress((idx + 1) / total)
        st.caption(f"Recommandation {idx + 1} sur {total}")
        st.divider()

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

        if show_explanation:
            st.markdown("#### 🤖 Pourquoi ce film vous est recommandé")
            st.info(st.session_state[f"expl_{idx}"])
            current_explanation = st.session_state[f"expl_{idx}"]
        else:
            current_explanation = ""

        st.divider()
        st.write("### Votre réaction à cette recommandation")

        fmt = lambda x: {1: "1 — Pas du tout", 2: "2", 3: "3 — Moyennement", 4: "4", 5: "5 — Tout à fait"}[x]

        st.write("**Compréhension perçue**")
        cp1 = st.select_slider("Je comprends pourquoi ce film m'a été recommandé.",
                               options=[1,2,3,4,5], value=3, key=f"cp1_{idx}", format_func=fmt)
        cp2 = st.select_slider("Je suis capable d'expliquer la logique de cette recommandation.",
                               options=[1,2,3,4,5], value=3, key=f"cp2_{idx}", format_func=fmt)

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

        st.write("**Confiance**")
        t1 = st.select_slider("Je fais confiance à cette recommandation.",
                               options=[1,2,3,4,5], value=3, key=f"t1_{idx}", format_func=fmt)
        t2 = st.select_slider("Ce film me semble pertinent pour moi.",
                               options=[1,2,3,4,5], value=3, key=f"t2_{idx}", format_func=fmt)
        t3 = st.select_slider("Ce système de recommandation me semble fiable.",
                               options=[1,2,3,4,5], value=3, key=f"t3_{idx}", format_func=fmt)

        st.divider()
        st.caption(f"Recommandation {idx + 1} sur {total}")

        if cr_display is None:
            st.warning(f"⬆️ Remontez en haut de page pour voir la recommandation {idx + 1} sur {total}, puis répondez à toutes les questions avant de continuer.")
        else:
            label = "Suivant →" if idx < total - 1 else "Terminer les recommandations →"
            clicked = st.button(label, type="primary", use_container_width=True)
            if clicked:
                with st.spinner("Chargement de la recommandation suivante en cours..." if idx < total - 1 else "Finalisation..."):
                    save_response({
                    "participant_id":                    st.session_state.participant_id,
                    "timestamp":                         datetime.now().isoformat(),
                    "film_numero":                       idx + 1,
                    "film_titre":                        film["title"],
                    "avec_explication":                  show_explanation,
                    "texte_explication":                 current_explanation,
                    "comprehension_percue_1":            cp1,
                    "comprehension_percue_2":            cp2,
                    "comprehension_reelle_bonne_reponse":st.session_state.get(f"qcm_correct_{idx}", ""),
                    "comprehension_reelle_reponse":      cr_display,
                    "comprehension_reelle_correcte":     real_correct,
                    "confiance_1":                       t1,
                    "confiance_2":                       t2,
                    "confiance_3":                       t3,
                    "films_aimes":                       " | ".join(st.session_state.liked_films)
                    })
                st.session_state.current_film_idx += 1
                st.rerun()

    # ── PROFIL DÉMOGRAPHIQUE ──────────────────────
    elif st.session_state.step == "profile":
        st.title("👤Votre profil")
        st.write("Dernière étape ! Quelques informations sur vous pour contextualiser les résultats de la recherche.")
        st.divider()

        st.markdown("**Votre tranche d'âge**")
        age = st.radio("", [
            "Moins de 18 ans", "18-24 ans", "25-34 ans",
            "35-44 ans", "45-54 ans", "55 ans et plus"
        ], index=None, key="age", horizontal=True)

        st.write("")
        st.markdown("**Votre situation professionnelle**")
        profession = st.radio("", [
            "Étudiant(e)",
            "Salarié(e) / Cadre",
            "Profession libérale / Indépendant(e)",
            "Enseignant(e) / Chercheur(e)",
            "Sans emploi",
            "Retraité(e)",
            "Autre"
        ], index=None, key="profession")

        st.write("")
        st.markdown("**Votre pays de résidence**")
        pays = st.radio("", [
            "France",
            "Autre pays européen",
            "Pays non européen"
        ], index=None, key="pays")

        st.divider()

        all_filled = all([age, profession, pays])

        if not all_filled:
            st.warning("Veuillez répondre à toutes les questions avant de continuer.")
        else:
            if st.button("Terminer l'étude ✓", type="primary", use_container_width=True):
                save_profile({
                    "participant_id":   st.session_state.participant_id,
                    "timestamp_profil": datetime.now().isoformat(),
                    "age":              age,
                    "profession":       profession,
                    "pays":             pays,
                })
                st.session_state.step = "finished"
                st.rerun()

    # ── FIN ──────────────────────────────────────
    elif st.session_state.step == "finished":
        st.balloons()
        st.success("🎉 Merci pour votre participation !")

        st.markdown("""
Vos réponses ont bien été enregistrées.

---

**🎙️ Vous souhaitez en savoir plus sur les algorithmes de recommandation ?**

Ce podcast de l'Université Paris 1 Panthéon-Sorbonne explore leur fonctionnement et leurs effets :
👉 [Les voies de l'IA — Radio France](https://www.radiofrance.fr/franceinfo/podcasts/les-voies-de-l-ia)

---

**📩 Vous souhaitez recevoir les résultats de cette étude une fois publiés ?**

N'hésitez pas à me contacter : **douaa.fredj@outlook.com**
        """)

        st.divider()
        st.markdown("💌 **Vous avez apprécié cette expérience ? Partagez-la avec votre entourage !**")
        st.write("Chaque participation supplémentaire contribue à la qualité de la recherche. Merci 🙏")

        app_url = "https://reco-system-experiment.streamlit.app"
        message = "J'ai participé à une étude scientifique sur les algorithmes de recommandation, menée par l'Université Paris 1 Panthéon-Sorbonne. N'hésitez pas à y contribuer vous aussi, votre participation enrichirait vraiment la recherche : "

        col1, col2, col3 = st.columns(3)
        with col1:
            whatsapp_url = f"https://wa.me/?text={requests.utils.quote(message + app_url)}"
            st.markdown(
                f'<a href="{whatsapp_url}" target="_blank"><button style="width:100%;background:#25D366;color:white;border:none;padding:10px;border-radius:8px;font-size:15px;cursor:pointer;">💬 WhatsApp</button></a>',
                unsafe_allow_html=True
            )
        with col2:
            email_url = f"mailto:?subject=Participe à cette étude !&body={requests.utils.quote(message + app_url)}"
            st.markdown(
                f'<a href="{email_url}"><button style="width:100%;background:#4A90D9;color:white;border:none;padding:10px;border-radius:8px;font-size:15px;cursor:pointer;">📧 E-mail</button></a>',
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f'<a href="https://t.me/share/url?url={requests.utils.quote(app_url)}&text={requests.utils.quote(message)}" target="_blank"><button style="width:100%;background:#229ED9;color:white;border:none;padding:10px;border-radius:8px;font-size:15px;cursor:pointer;">✈️ Telegram</button></a>',
                unsafe_allow_html=True
            )

        st.divider()
        if st.button("Nouvelle participation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()