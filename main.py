import streamlit as st
import tensorflow as tf
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; height: 0 !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 1.5rem !important; }

/* ── App background ── */
.stApp {
    background: radial-gradient(ellipse at top left, #0b1f0e 0%, #060e08 60%, #020904 100%);
    min-height: 100vh;
    color: #dde8d8;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

.sidebar-logo {
    font-size: 1.5rem;
    font-weight: 800;
    color: #4ade80;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(74,222,128,0.25), transparent);
    margin: 0.8rem 0 1.2rem 0;
}
.sidebar-stat {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.45rem 0;
    color: #7ab87a;
    font-size: 0.82rem;
    font-weight: 500;
}
.sidebar-stat .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    flex-shrink: 0;
}

/* ── Sidebar selectbox ── */
[data-testid="stSidebar"] .stSelectbox label {
    color: #7ab87a !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: rgba(74,222,128,0.07) !important;
    border: 1px solid rgba(74,222,128,0.25) !important;
    border-radius: 10px !important;
    color: #c8e6c0 !important;
}

/* ── Page hero ── */
.hero {
    position: relative;
    background: linear-gradient(135deg, #052e10 0%, #083618 60%, #05200c 100%);
    border: 1px solid rgba(74,222,128,0.18);
    border-radius: 22px;
    padding: 2.2rem 2.5rem 2.2rem 2.5rem;
    margin-bottom: 1.8rem;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(74,222,128,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6ee7a0 0%, #4ade80 50%, #86efac 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero p { color: #86efac; font-size: 1rem; opacity: 0.8; margin: 0; }

/* ── Section label ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4ade80;
    margin: 1.6rem 0 1rem 0;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(74,222,128,0.18);
}

/* ── Feature cards ── */
.fcard {
    background: rgba(5,46,16,0.45);
    border: 1px solid rgba(74,222,128,0.13);
    border-radius: 18px;
    padding: 1.6rem 1.3rem 1.4rem;
    text-align: center;
    height: 100%;
    transition: border-color 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease;
}
.fcard:hover {
    border-color: rgba(74,222,128,0.38);
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.fcard .ico { font-size: 2.4rem; margin-bottom: 0.75rem; }
.fcard h3 { color: #4ade80; font-size: 0.98rem; font-weight: 700; margin: 0 0 0.5rem; }
.fcard p  { color: #90b890; font-size: 0.84rem; line-height: 1.6; margin: 0; }

/* ── How-it-works steps ── */
.step {
    background: rgba(5,46,16,0.3);
    border-left: 3px solid #22c55e;
    border-radius: 0 14px 14px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.step .num {
    background: rgba(74,222,128,0.15);
    color: #4ade80;
    border-radius: 8px;
    min-width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 800;
    flex-shrink: 0;
}
.step .txt { color: #a8c4a0; font-size: 0.9rem; line-height: 1.55; padding-top: 0.35rem; }
.step .txt b { color: #c8e6c0; }

/* ── Stat cards (About) ── */
.scard {
    background: rgba(5,46,16,0.5);
    border: 1px solid rgba(74,222,128,0.16);
    border-radius: 18px;
    padding: 1.5rem 1rem;
    text-align: center;
}
.scard .val {
    font-size: 2rem;
    font-weight: 800;
    color: #4ade80;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}
.scard .lbl { color: #7ab87a; font-size: 0.78rem; margin-top: 0.4rem; font-weight: 500; letter-spacing: 0.05em; }

/* ── Info block ── */
.info-block {
    background: rgba(5,46,16,0.4);
    border: 1px solid rgba(74,222,128,0.14);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    color: #a8c4a0;
    font-size: 0.88rem;
    line-height: 1.7;
}
.info-block b, .info-block strong { color: #c8e6c0; }
.info-block code {
    background: rgba(74,222,128,0.12);
    color: #86efac;
    padding: 1px 6px;
    border-radius: 5px;
    font-size: 0.83rem;
}

/* ── File uploader — force dark theme ── */
[data-testid="stFileUploader"] {
    background: rgba(5,46,16,0.35) !important;
    border: 2px dashed rgba(74,222,128,0.3) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"] label {
    color: #7ab87a !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(5,46,16,0.5) !important;
    border: none !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #7ab87a !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(74,222,128,0.12) !important;
    color: #4ade80 !important;
    border: 1px solid rgba(74,222,128,0.3) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"] svg { fill: #4ade80 !important; }
[data-testid="stFileUploaderDropzone"] small { color: #5a8a5a !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(74,222,128,0.2) !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #15803d 0%, #166534 100%) !important;
    box-shadow: 0 8px 24px rgba(74,222,128,0.32) !important;
    transform: translateY(-1px) !important;
}

/* ── Image ── */
[data-testid="stImage"] { border-radius: 14px; overflow: hidden; }
[data-testid="stImage"] img { border-radius: 14px !important; }
.stImage > div { text-align: center; }

/* ── Empty state ── */
.empty-state {
    background: rgba(5,46,16,0.3);
    border: 1px dashed rgba(74,222,128,0.18);
    border-radius: 18px;
    padding: 3rem 2rem;
    text-align: center;
    color: #5a8a5a;
}
.empty-state .big { font-size: 3rem; margin-bottom: 0.8rem; }
.empty-state .title { color: #4ade80; font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem; }

/* ── Result card ── */
.result-card {
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-top: 1.2rem;
}
.result-card.healthy {
    background: linear-gradient(135deg, rgba(5,46,16,0.8) 0%, rgba(6,78,26,0.55) 100%);
    border: 1px solid rgba(74,222,128,0.35);
}
.result-card.disease {
    background: linear-gradient(135deg, rgba(46,25,5,0.85) 0%, rgba(78,50,6,0.55) 100%);
    border: 1px solid rgba(251,191,36,0.35);
}
.result-card .r-icon { font-size: 3.2rem; margin-bottom: 0.7rem; }
.result-card .r-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.85rem;
    border-radius: 50px;
    margin-bottom: 0.8rem;
}
.result-card.healthy .r-badge { background: rgba(74,222,128,0.15); color: #86efac; }
.result-card.disease  .r-badge { background: rgba(251,191,36,0.15); color: #fde68a; }
.result-card .r-name {
    font-size: 1.45rem;
    font-weight: 800;
    line-height: 1.3;
}
.result-card.healthy .r-name { color: #4ade80; }
.result-card.disease  .r-name { color: #fbbf24; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #4ade80 !important; }

/* ── Disease info panel ── */
.dinfo {
    background: rgba(5,46,16,0.4);
    border: 1px solid rgba(74,222,128,0.16);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.4rem;
}
.dinfo h3 {
    font-size: 1rem;
    font-weight: 700;
    color: #4ade80;
    margin: 0 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.dinfo p {
    color: #a8c4a0;
    font-size: 0.88rem;
    line-height: 1.7;
    margin: 0 0 0.6rem 0;
}
.dinfo ul {
    margin: 0;
    padding-left: 1.2rem;
}
.dinfo ul li {
    color: #90b890;
    font-size: 0.86rem;
    line-height: 1.65;
    margin-bottom: 0.3rem;
}
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.pill {
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.22);
    border-radius: 50px;
    padding: 0.25rem 0.8rem;
    font-size: 0.78rem;
    color: #86efac;
    font-weight: 500;
}
.pill.red {
    background: rgba(251,191,36,0.1);
    border-color: rgba(251,191,36,0.25);
    color: #fde68a;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #030804; }
::-webkit-scrollbar-thumb { background: #1a4d28; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4ade80; }

/* ── Pills Navigation ── */
[data-testid="stPills"] {
    justify-content: center;
    margin-bottom: 0.5rem;
}
[data-testid="stPills"] button {
    background: rgba(5,46,16,0.3) !important;
    border: 1px solid rgba(74,222,128,0.25) !important;
    border-radius: 12px !important;
    color: #7ab87a !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stPills"] button:hover {
    border-color: #4ade80 !important;
    color: #4ade80 !important;
    background: rgba(74,222,128,0.1) !important;
}
[data-testid="stPills"] button[data-state="active"], [data-testid="stPills"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(74,222,128,0.3) !important;
}

/* ── Responsive Mobile Adjustments ── */
@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem !important; }
    .hero { padding: 1.5rem 1rem !important; text-align: center; }
    .hero h1 { font-size: 1.6rem; }
    .hero p { font-size: 0.9rem; }
    .sidebar-logo { justify-content: center; font-size: 1.3rem; }
    .sidebar-stat { font-size: 0.75rem; }
    .result-card .r-icon { font-size: 2.5rem; }
    .result-card .r-name { font-size: 1.2rem; }
    .fcard { margin-bottom: 1rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model('trained_model.h5')

def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return int(np.argmax(prediction))

# ── Disease info database (no API needed) ────────────────────────────────────
DISEASE_INFO = {
    # ── Apple ──────────────────────────────────────────────────────────────
    'Apple — Apple Scab': {
        'emoji': '🍎',
        'description': 'Apple Scab is a fungal disease caused by Venturia inaequalis. It produces olive-green to brown scab-like lesions on leaves and fruit, leading to premature leaf drop and disfigured, cracked fruit.',
        'symptoms': ['Olive-green or brown spots on leaves', 'Scab-like lesions on fruit skin', 'Premature leaf and fruit drop', 'Stunted growth in severe cases'],
        'treatment': ['Apply fungicides (captan, mancozeb) at bud break', 'Remove and destroy infected leaves and fruit', 'Prune trees to improve air circulation', 'Plant resistant apple varieties', 'Avoid overhead irrigation'],
    },
    'Apple — Black Rot': {
        'emoji': '🍎',
        'description': 'Black Rot is a fungal disease caused by Botryosphaeria obtusa. It affects leaves, fruit, and bark, causing frog-eye leaf spots and mummified black fruit.',
        'symptoms': ['Circular purple spots with brown centers on leaves', 'Black, rotting fruit that shrivel and mummify', 'Cankers on branches', 'Yellowing and early leaf drop'],
        'treatment': ['Prune out dead or cankered wood', 'Remove mummified fruit from trees and ground', 'Apply copper-based or captan fungicide', 'Maintain proper tree nutrition', 'Avoid wounding bark during pruning'],
    },
    'Apple — Cedar Apple Rust': {
        'emoji': '🍎',
        'description': 'Cedar Apple Rust is caused by Gymnosporangium juniperi-virginianae, a fungus requiring two hosts — apple and eastern red cedar. It produces bright orange-yellow spots on apple leaves.',
        'symptoms': ['Bright orange-yellow spots on upper leaf surface', 'Tube-like structures on leaf undersides', 'Premature defoliation', 'Distorted and stunted fruit'],
        'treatment': ['Apply fungicides (myclobutanil, triadimefon) early in season', 'Remove nearby juniper/cedar trees if possible', 'Plant rust-resistant apple cultivars', 'Rake and dispose of fallen leaves'],
    },
    'Apple — Healthy': {
        'emoji': '✅',
        'description': 'Your apple plant appears healthy! Healthy apple leaves are deep green, smooth, and show no spots, lesions, or discoloration.',
        'symptoms': ['Deep green, smooth leaves', 'No visible spots or lesions', 'Normal fruit development'],
        'treatment': ['Continue regular watering and fertilization', 'Prune annually for good air circulation', 'Monitor regularly for early signs of disease', 'Apply preventive sprays during high-humidity periods'],
    },
    # ── Blueberry ──────────────────────────────────────────────────────────
    'Blueberry — Healthy': {
        'emoji': '✅',
        'description': 'Your blueberry plant is healthy! Healthy blueberry plants have vibrant green leaves and produce plump berries.',
        'symptoms': ['Vibrant green foliage', 'No spots or wilting', 'Healthy berry formation'],
        'treatment': ['Maintain soil pH between 4.5–5.5', 'Apply balanced fertilizer in spring', 'Mulch to retain moisture and suppress weeds', 'Prune old canes every year'],
    },
    # ── Cherry ─────────────────────────────────────────────────────────────
    'Cherry — Powdery Mildew': {
        'emoji': '🍒',
        'description': 'Cherry Powdery Mildew is caused by Podosphaera clandestina. A white powdery coating covers leaves, shoot tips, and fruit, reducing photosynthesis and causing leaf curling.',
        'symptoms': ['White powdery coating on leaves and shoots', 'Curling and twisting of young leaves', 'Stunted shoot growth', 'Reduced fruit quality'],
        'treatment': ['Apply sulfur-based or potassium bicarbonate fungicides', 'Remove and destroy infected plant parts', 'Improve air circulation by pruning', 'Avoid excessive nitrogen fertilization', 'Plant mildew-resistant cherry varieties'],
    },
    'Cherry — Healthy': {
        'emoji': '✅',
        'description': 'Your cherry tree is healthy! Healthy cherry leaves are glossy green with no spots or powdery coating.',
        'symptoms': ['Glossy green leaves', 'No powdery coating', 'Vigorous new growth'],
        'treatment': ['Water at the base, avoid wetting foliage', 'Fertilize in early spring', 'Prune for light penetration', 'Monitor for pests and diseases seasonally'],
    },
    # ── Corn ───────────────────────────────────────────────────────────────
    'Corn — Cercospora / Gray Leaf Spot': {
        'emoji': '🌽',
        'description': 'Gray Leaf Spot, caused by Cercospora zeae-maydis, is a major fungal disease of corn producing rectangular grey-tan lesions aligned with leaf veins, leading to significant yield loss.',
        'symptoms': ['Rectangular, tan to gray lesions on leaves', 'Lesions run parallel to leaf veins', 'Lesions may merge causing whole leaf blight', 'Premature plant death in severe cases'],
        'treatment': ['Apply foliar fungicides (strobilurins, triazoles)', 'Plant resistant hybrids', 'Rotate crops — avoid continuous corn', 'Till to reduce infected residue', 'Improve field drainage'],
    },
    'Corn — Common Rust': {
        'emoji': '🌽',
        'description': 'Common Rust is caused by Puccinia sorghi. It produces brick-red to brown pustules on both leaf surfaces, reducing photosynthetic area and yield.',
        'symptoms': ['Circular to elongated brick-red pustules on leaves', 'Pustules on both upper and lower leaf surfaces', 'Yellowing of surrounding tissue', 'Heavy infection causes entire leaf necrosis'],
        'treatment': ['Apply fungicides containing triazoles or strobilurins', 'Plant rust-resistant corn hybrids', 'Early planting to avoid peak rust season', 'Scout fields regularly for early detection'],
    },
    'Corn — Northern Leaf Blight': {
        'emoji': '🌽',
        'description': 'Northern Leaf Blight is caused by Exserohilum turcicum fungi. It creates large, cigar-shaped grey-green lesions on corn leaves that turn tan as they mature.',
        'symptoms': ['Long cigar-shaped tan or gray-green lesions', 'Lesions may be 2.5–15 cm long', 'Dark sporulation within lesions', 'Rapid spread in cool, humid conditions'],
        'treatment': ['Apply strobilurin or triazole fungicides', 'Plant resistant hybrids', 'Practice crop rotation', 'Bury or till infected residue', 'Ensure good field drainage'],
    },
    'Corn — Healthy': {
        'emoji': '✅',
        'description': 'Your corn plant is healthy! Healthy corn has bright green leaves with no lesions, pustules, or blight symptoms.',
        'symptoms': ['Bright green uniform leaves', 'No rust pustules or tan lesions', 'Vigorous growth'],
        'treatment': ['Maintain balanced fertilization (especially nitrogen)', 'Ensure adequate irrigation', 'Scout regularly during growing season', 'Practice crop rotation'],
    },
    # ── Grape ──────────────────────────────────────────────────────────────
    'Grape — Black Rot': {
        'emoji': '🍇',
        'description': 'Grape Black Rot is caused by Guignardia bidwellii. It attacks all green parts of the vine, causing tan leaf lesions and black, shriveled mummified berries.',
        'symptoms': ['Tan lesions with dark borders on leaves', 'Black, shriveled mummified berries', 'Dark lesions on shoots and stems', 'Small black dots (pycnidia) in lesions'],
        'treatment': ['Apply fungicides (myclobutanil, mancozeb) starting at budbreak', 'Remove mummified berries and infected debris', 'Prune to open the canopy for air flow', 'Avoid overhead irrigation'],
    },
    'Grape — Esca (Black Measles)': {
        'emoji': '🍇',
        'description': 'Esca is a complex wood disease of grapevines caused by multiple fungi including Phaeomoniella chlamydospora. It causes tiger-stripe leaf patterns and internal wood decay.',
        'symptoms': ['Tiger-stripe yellowing or reddening on leaves', 'Dark spots on berries ("measles")', 'Sudden wilting of shoots', 'Internal wood discoloration and decay'],
        'treatment': ['No chemical cure exists; remove and destroy infected vines', 'Make clean pruning cuts and seal with wound sealant', 'Avoid large pruning wounds', 'Replant with certified healthy stock', 'Use double-pruning technique to reduce infection'],
    },
    'Grape — Leaf Blight': {
        'emoji': '🍇',
        'description': 'Grape Leaf Blight (Isariopsis Leaf Spot) is caused by Pseudocercospora vitis. Dark angular spots appear on leaves, leading to defoliation and reduced berry quality.',
        'symptoms': ['Dark brown angular spots on leaves', 'Spots bordered by veins', 'Yellowing around lesions', 'Premature leaf drop'],
        'treatment': ['Apply copper-based or mancozeb fungicides', 'Prune affected parts promptly', 'Improve canopy air circulation', 'Collect and destroy fallen leaves'],
    },
    'Grape — Healthy': {
        'emoji': '✅',
        'description': 'Your grapevine is healthy! Healthy grape leaves are bright green with no spots, stripes, or discoloration.',
        'symptoms': ['Bright green, unblemished leaves', 'Vigorous cane growth', 'Normal berry development'],
        'treatment': ['Prune annually to maintain open canopy', 'Apply preventive fungicide program', 'Monitor soil moisture and drainage', 'Fertilize based on soil test results'],
    },
    # ── Orange ─────────────────────────────────────────────────────────────
    'Orange — Huanglongbing (Citrus Greening)': {
        'emoji': '🍊',
        'description': 'Huanglongbing (HLB) or Citrus Greening is caused by Candidatus Liberibacter bacteria spread by the Asian citrus psyllid. It is one of the most devastating citrus diseases worldwide — currently incurable.',
        'symptoms': ['Asymmetric yellowing of leaves (blotchy mottle)', 'Small, lopsided, bitter fruit', 'Premature fruit drop', 'Twig dieback and decline'],
        'treatment': ['No cure exists — remove and destroy infected trees', 'Control Asian citrus psyllid with insecticides', 'Plant certified disease-free nursery stock', 'Implement strict quarantine measures', 'Use reflective mulches to repel psyllids'],
    },
    # ── Peach ──────────────────────────────────────────────────────────────
    'Peach — Bacterial Spot': {
        'emoji': '🍑',
        'description': 'Peach Bacterial Spot is caused by Xanthomonas arboricola pv. pruni. It produces water-soaked lesions on leaves, sunken spots on fruit, and cankers on twigs.',
        'symptoms': ['Water-soaked angular spots on leaves', 'Spots turn purple-brown and drop out (shot-hole effect)', 'Sunken dark spots on fruit skin', 'Twig cankers in severe infection'],
        'treatment': ['Apply copper bactericides during dormancy and early season', 'Plant resistant peach varieties', 'Avoid overhead irrigation', 'Prune to improve canopy airflow', 'Avoid excessive nitrogen which promotes susceptible growth'],
    },
    'Peach — Healthy': {
        'emoji': '✅',
        'description': 'Your peach tree is healthy! Healthy peach leaves are long, lance-shaped, and deep green with no spots or lesions.',
        'symptoms': ['Lance-shaped deep green leaves', 'No spots or cankers', 'Vigorous shoot growth'],
        'treatment': ['Apply dormant copper spray each winter', 'Thin fruit to improve size and air circulation', 'Monitor for borers and aphids', 'Fertilize in early spring'],
    },
    # ── Pepper Bell ────────────────────────────────────────────────────────
    'Pepper Bell — Bacterial Spot': {
        'emoji': '🫑',
        'description': 'Bacterial Spot of pepper is caused by Xanthomonas euvesicatoria. It produces water-soaked, scab-like spots on leaves and fruit, severely reducing marketability.',
        'symptoms': ['Small water-soaked lesions turning dark brown', 'Lesions with yellow halos on leaves', 'Raised, scab-like spots on fruit', 'Severe defoliation in wet conditions'],
        'treatment': ['Apply copper-based bactericides preventively', 'Use certified disease-free seed', 'Avoid working in fields when wet', 'Rotate crops — avoid solanaceous crops for 2–3 years', 'Plant resistant pepper varieties'],
    },
    'Pepper Bell — Healthy': {
        'emoji': '✅',
        'description': 'Your bell pepper plant is healthy! Healthy pepper plants have bright green glossy leaves and firm, unblemished fruit.',
        'symptoms': ['Glossy green leaves', 'No water-soaked lesions', 'Firm, smooth fruit development'],
        'treatment': ['Water consistently to avoid blossom end rot', 'Fertilize with balanced NPK', 'Stake plants to prevent stem breakage', 'Monitor for aphids and spider mites'],
    },
    # ── Potato ─────────────────────────────────────────────────────────────
    'Potato — Early Blight': {
        'emoji': '🥔',
        'description': 'Potato Early Blight is caused by Alternaria solani. It produces characteristic target-like concentric ring spots on lower leaves first, then spreads upward.',
        'symptoms': ['Dark brown spots with concentric rings (target pattern)', 'Yellow halo around lesions', 'Starts on older lower leaves', 'Defoliation and reduced tuber yield'],
        'treatment': ['Apply fungicides (chlorothalonil, mancozeb, azoxystrobin)', 'Remove infected lower leaves early', 'Avoid overhead irrigation', 'Practice crop rotation (3-year cycle)', 'Plant certified disease-free seed potatoes'],
    },
    'Potato — Late Blight': {
        'emoji': '🥔',
        'description': 'Potato Late Blight is caused by Phytophthora infestans — the same pathogen responsible for the Irish potato famine. It is highly destructive and spreads rapidly in cool, wet conditions.',
        'symptoms': ['Pale green to dark water-soaked lesions on leaves', 'White fluffy mold on leaf undersides', 'Brown-black rotting of stems', 'Rapid plant collapse in humid weather'],
        'treatment': ['Apply fungicides (metalaxyl, chlorothalonil) immediately', 'Destroy heavily infected plants by burning', 'Improve field drainage', 'Plant resistant varieties', 'Harvest tubers promptly in infected fields'],
    },
    'Potato — Healthy': {
        'emoji': '✅',
        'description': 'Your potato plant is healthy! Healthy potato plants have dark green compound leaves with no spots, mold, or water-soaked areas.',
        'symptoms': ['Dark green compound leaves', 'Vigorous upright stems', 'No lesions or wilting'],
        'treatment': ['Hill soil around plants as they grow', 'Maintain consistent soil moisture', 'Apply balanced fertilizer at planting', 'Scout for Colorado potato beetles'],
    },
    # ── Raspberry ──────────────────────────────────────────────────────────
    'Raspberry — Healthy': {
        'emoji': '✅',
        'description': 'Your raspberry plant is healthy! Healthy raspberry canes are vigorous with deep green leaves and no rust, spots, or lesions.',
        'symptoms': ['Deep green compound leaves', 'Firm canes without lesions', 'Healthy fruit development'],
        'treatment': ['Prune old floricanes after fruiting', 'Support canes with trellis', 'Maintain soil pH 5.5–6.5', 'Apply mulch to suppress weeds and retain moisture'],
    },
    # ── Soybean ────────────────────────────────────────────────────────────
    'Soybean — Healthy': {
        'emoji': '✅',
        'description': 'Your soybean plant is healthy! Healthy soybeans have trifoliate green leaves with no spots, rust, or yellowing.',
        'symptoms': ['Trifoliate bright green leaves', 'No rust pustules or lesions', 'Vigorous pod formation'],
        'treatment': ['Practice crop rotation with non-legumes', 'Inoculate seeds with Bradyrhizobium for nitrogen fixation', 'Scout for soybean aphid and spider mites', 'Avoid compaction to preserve root health'],
    },
    # ── Squash ─────────────────────────────────────────────────────────────
    'Squash — Powdery Mildew': {
        'emoji': '🎃',
        'description': 'Squash Powdery Mildew is caused by Podosphaera xanthii. A white powdery fungal growth covers leaves, reducing photosynthesis, causing premature senescence and poor fruit production.',
        'symptoms': ['White powdery spots on upper leaf surface', 'Yellowing of infected leaves', 'Leaf curling and browning', 'Early plant death in severe cases'],
        'treatment': ['Apply potassium bicarbonate, sulfur, or neem oil sprays', 'Space plants widely for air circulation', 'Avoid overhead watering', 'Remove badly infected leaves', 'Plant mildew-resistant varieties'],
    },
    # ── Strawberry ─────────────────────────────────────────────────────────
    'Strawberry — Leaf Scorch': {
        'emoji': '🍓',
        'description': 'Strawberry Leaf Scorch is caused by Diplocarpon earlianum. It produces small, irregular purple spots on leaf surfaces that expand and merge, scorching the leaf.',
        'symptoms': ['Small irregular purple/red spots on leaves', 'Spots coalesce giving a scorched appearance', 'Leaf margins turn brown', 'Severe defoliation in wet conditions'],
        'treatment': ['Apply fungicides (captan, thiram) preventively', 'Remove and destroy infected leaves', 'Improve air circulation by proper spacing', 'Avoid overhead irrigation', 'Renovate planting after harvest'],
    },
    'Strawberry — Healthy': {
        'emoji': '✅',
        'description': 'Your strawberry plant is healthy! Healthy strawberry plants have bright green trifoliate leaves with no spots, burn marks, or powdery coating.',
        'symptoms': ['Bright green trifoliate leaves', 'No lesions or discoloration', 'Healthy runner production'],
        'treatment': ['Renovate beds after harvest to reduce disease', 'Mulch with straw to prevent soil splash', 'Maintain pH 6.0–6.5', 'Remove old foliage in autumn'],
    },
    # ── Tomato ─────────────────────────────────────────────────────────────
    'Tomato — Bacterial Spot': {
        'emoji': '🍅',
        'description': 'Tomato Bacterial Spot is caused by Xanthomonas species. Water-soaked lesions appear on leaves and fruit, reducing plant vigor and making fruit unmarketable.',
        'symptoms': ['Small water-soaked circular spots on leaves', 'Spots with yellow halos', 'Raised, scab-like lesions on fruit', 'Defoliation in severe infections'],
        'treatment': ['Apply copper bactericide + mancozeb sprays', 'Use certified disease-free transplants', 'Avoid working in wet fields', 'Rotate crops for 2–3 years away from tomatoes', 'Remove and destroy infected plant debris'],
    },
    'Tomato — Early Blight': {
        'emoji': '🍅',
        'description': 'Tomato Early Blight caused by Alternaria solani produces distinctive bullseye-patterned lesions starting on older leaves at the bottom of the plant.',
        'symptoms': ['Dark concentric ring (bullseye) spots on leaves', 'Yellow halo around lesions', 'Starts on older lower leaves and moves up', 'Dark stem lesions at soil line (collar rot)'],
        'treatment': ['Apply fungicides (chlorothalonil, mancozeb) every 7–10 days', 'Remove infected lower leaves', 'Stake plants for better air circulation', 'Avoid overhead watering', 'Mulch to prevent soil splash-up'],
    },
    'Tomato — Late Blight': {
        'emoji': '🍅',
        'description': 'Tomato Late Blight, caused by Phytophthora infestans, is a devastating disease causing rapid collapse of plants in cool, wet weather. It ruins fruit and can destroy entire crops.',
        'symptoms': ['Large, irregularly shaped water-soaked lesions', 'White mold on leaf undersides', 'Brown-black stem rot', 'Firm, brown greasy rot on fruit'],
        'treatment': ['Apply systemic fungicides (metalaxyl-M) immediately', 'Remove and bag infected plants — do not compost', 'Improve drainage and air circulation', 'Avoid overhead irrigation', 'Plant resistant varieties'],
    },
    'Tomato — Leaf Mold': {
        'emoji': '🍅',
        'description': 'Tomato Leaf Mold is caused by Passalora fulva (Cladosporium fulvum). It produces pale green-yellow spots on upper leaf surfaces and olive-green to grey mold on the undersides.',
        'symptoms': ['Pale yellow spots on upper leaf surface', 'Olive-green or grey mold on leaf undersides', 'Leaves curl upward and die', 'Rarely affects fruit but reduces yield significantly'],
        'treatment': ['Improve greenhouse ventilation', 'Apply fungicides (chlorothalonil, mancozeb)', 'Reduce relative humidity below 85%', 'Remove infected leaves promptly', 'Plant resistant varieties'],
    },
    'Tomato — Septoria Leaf Spot': {
        'emoji': '🍅',
        'description': 'Septoria Leaf Spot, caused by Septoria lycopersici, produces many small, water-soaked spots with dark edges and light grey centers on lower leaves, leading to severe defoliation.',
        'symptoms': ['Small circular spots with dark borders and grey centers', 'Tiny black dots (pycnidia) visible in lesion centers', 'Starts on lower leaves after fruit set', 'Progressive upward defoliation'],
        'treatment': ['Apply fungicides (chlorothalonil, copper, mancozeb)', 'Remove infected leaves below the first flower cluster', 'Mulch to prevent soil splash', 'Avoid overhead irrigation', 'Rotate crops annually'],
    },
    'Tomato — Spider Mites': {
        'emoji': '🍅',
        'description': 'Two-spotted spider mites (Tetranychus urticae) are tiny arachnids that feed on plant cells, causing bronze stippling, leaf discoloration, and fine webbing on the undersides of leaves.',
        'symptoms': ['Yellow or bronze stippling on leaf surfaces', 'Fine silken webbing on leaf undersides', 'Leaves turn yellow, dry out and drop', 'Worse in hot, dry conditions'],
        'treatment': ['Apply miticides (abamectin, bifenazate) or neem oil', 'Spray water on leaves to knock off mites', 'Introduce predatory mites (Phytoseiidae)', 'Increase humidity around plants', 'Avoid broad-spectrum insecticides that kill natural predators'],
    },
    'Tomato — Target Spot': {
        'emoji': '🍅',
        'description': 'Target Spot is caused by Corynespora cassiicola. It produces circular, target-patterned brown lesions on leaves and fruit, causing defoliation and fruit rot.',
        'symptoms': ['Circular brown lesions with concentric rings', 'Dark margins around spots', 'Sunken dark lesions on fruit', 'Severe defoliation in humid conditions'],
        'treatment': ['Apply fungicides (azoxystrobin, chlorothalonil)', 'Remove infected leaves promptly', 'Stake plants for good air circulation', 'Avoid overhead irrigation', 'Rotate with non-solanaceous crops'],
    },
    'Tomato — Yellow Leaf Curl Virus': {
        'emoji': '🍅',
        'description': 'Tomato Yellow Leaf Curl Virus (TYLCV) is a whitefly-transmitted begomovirus. Infected plants show severe stunting, upward leaf curling, and chlorosis — no chemical cure exists.',
        'symptoms': ['Leaf edges curl upward and inward', 'Yellowing (chlorosis) of leaves', 'Severe stunting of new growth', 'Flower drop and very low fruit set'],
        'treatment': ['Control whitefly vectors with insecticides (imidacloprid)', 'Use reflective silver mulches to repel whiteflies', 'Install insect-proof screens in greenhouse', 'Remove and destroy infected plants early', 'Plant TYLCV-resistant tomato varieties'],
    },
    'Tomato — Mosaic Virus': {
        'emoji': '🍅',
        'description': 'Tomato Mosaic Virus (ToMV) causes mottled green-yellow mosaic patterns on leaves, distortion, and reduced fruit size. It spreads through contact with infected plant sap.',
        'symptoms': ['Mosaic pattern of light and dark green on leaves', 'Leaf distortion and blistering', 'Stunted plant growth', 'Small, poorly colored fruit'],
        'treatment': ['No chemical cure — remove and destroy infected plants', 'Sanitize tools with 1:9 bleach solution frequently', 'Wash hands before handling plants', 'Plant TMV-resistant varieties', 'Control aphids that spread the virus'],
    },
    'Tomato — Healthy': {
        'emoji': '✅',
        'description': 'Your tomato plant is healthy! Healthy tomato plants have rich green compound leaves with no spots, mold, mosaic patterns, or curling.',
        'symptoms': ['Rich green compound leaves', 'No lesions, mold, or discoloration', 'Strong upright stems with healthy flower set'],
        'treatment': ['Water deeply at the base 2–3 times per week', 'Apply calcium to prevent blossom end rot', 'Stake or cage plants for support', 'Fertilize with phosphorus-rich fertilizer at transplant'],
    },
}

CLASS_NAMES = [
    'Apple — Apple Scab', 'Apple — Black Rot', 'Apple — Cedar Apple Rust', 'Apple — Healthy',
    'Blueberry — Healthy',
    'Cherry — Powdery Mildew', 'Cherry — Healthy',
    'Corn — Cercospora / Gray Leaf Spot', 'Corn — Common Rust',
    'Corn — Northern Leaf Blight', 'Corn — Healthy',
    'Grape — Black Rot', 'Grape — Esca (Black Measles)',
    'Grape — Leaf Blight', 'Grape — Healthy',
    'Orange — Huanglongbing (Citrus Greening)',
    'Peach — Bacterial Spot', 'Peach — Healthy',
    'Pepper Bell — Bacterial Spot', 'Pepper Bell — Healthy',
    'Potato — Early Blight', 'Potato — Late Blight', 'Potato — Healthy',
    'Raspberry — Healthy', 'Soybean — Healthy', 'Squash — Powdery Mildew',
    'Strawberry — Leaf Scorch', 'Strawberry — Healthy',
    'Tomato — Bacterial Spot', 'Tomato — Early Blight', 'Tomato — Late Blight',
    'Tomato — Leaf Mold', 'Tomato — Septoria Leaf Spot', 'Tomato — Spider Mites',
    'Tomato — Target Spot', 'Tomato — Yellow Leaf Curl Virus',
    'Tomato — Mosaic Virus', 'Tomato — Healthy',
]

# ── Navigation (UI Pills) ─────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = "🏠  Home"

st.markdown('<div class="sidebar-logo">🌿 PlantGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# Use st.pills for native responsive horizontal wrapping navigation
selected_page = st.pills(
    "Navigation", 
    ["🏠  Home", "ℹ️  About", "🔬  Disease Recognition"], 
    selection_mode="single",
    label_visibility="collapsed",
    default=st.session_state.page
)

# Only update if a valid new selection is made (prevent deselecting to hide app)
if selected_page and selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

page = st.session_state.page

st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; justify-content: center;">
    <div class="sidebar-stat"><span class="dot"></span>38 disease classes</div>
    <div class="sidebar-stat"><span class="dot"></span>14 crop varieties</div>
    <div class="sidebar-stat"><span class="dot"></span>87K+ training images</div>
    <div class="sidebar-stat"><span class="dot"></span>CNN · TensorFlow</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if "Home" in page:

    # Hero
    st.markdown("""
    <div class="hero">
      <h1>🌿 PlantGuard AI</h1>
      <p>Instant plant disease detection — powered by deep learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Hero image
    st.image("home_page.jpeg", use_container_width=True)

    # Feature cards
    st.markdown('<div class="section-label">✦ Why PlantGuard AI</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="fcard">
            <div class="ico">🎯</div>
            <h3>High Accuracy</h3>
            <p>State-of-the-art CNN trained on 87K+ images across 38 disease classes.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="fcard">
            <div class="ico">⚡</div>
            <h3>Instant Results</h3>
            <p>Get your leaf diagnosis in seconds — no waiting, no manual analysis.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="fcard">
            <div class="ico">🌱</div>
            <h3>14 Crop Types</h3>
            <p>From tomatoes and potatoes to grapes, corn, apples and more.</p>
        </div>""", unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="section-label">🚀 How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step">
      <div class="num">1</div>
      <div class="txt"><b>Upload a leaf photo</b> — Navigate to Disease Recognition and upload a clear image of a plant leaf.</div>
    </div>
    <div class="step">
      <div class="num">2</div>
      <div class="txt"><b>AI analysis</b> — Our CNN model processes the image at 128×128 and detects disease patterns.</div>
    </div>
    <div class="step">
      <div class="num">3</div>
      <div class="txt"><b>Instant result</b> — See the predicted disease class with healthy/diseased status at a glance.</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif "About" in page:

    st.markdown("""
    <div class="hero">
      <h1>ℹ️ About</h1>
      <p>Dataset overview, model information and architecture</p>
    </div>
    """, unsafe_allow_html=True)

    # Top stats
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown('<div class="scard"><div class="val">87K+</div><div class="lbl">Training Images</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="scard"><div class="val">38</div><div class="lbl">Disease Classes</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="scard"><div class="val">80/20</div><div class="lbl">Train / Val Split</div></div>', unsafe_allow_html=True)

    # Dataset details
    st.markdown('<div class="section-label">📊 Dataset Details</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
        This dataset was recreated using offline augmentation from the original <b>PlantVillage</b> dataset.
        It consists of approximately <b>87,000 RGB images</b> of healthy and diseased crop leaves categorised
        into <b>38 distinct classes</b>. Images are split 80/20 into training and validation sets, preserving
        the original directory structure. A separate test set of <b>33 images</b> is included for final evaluation.
    </div>
    """, unsafe_allow_html=True)

    # Split breakdown
    st.markdown('<div class="section-label">📁 Dataset Split</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        st.markdown('<div class="scard"><div class="val">70,295</div><div class="lbl">Train Images</div></div>', unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="scard"><div class="val">17,572</div><div class="lbl">Validation Images</div></div>', unsafe_allow_html=True)
    with d3:
        st.markdown('<div class="scard"><div class="val">33</div><div class="lbl">Test Images</div></div>', unsafe_allow_html=True)

    # Model info
    st.markdown('<div class="section-label">🧠 Model Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
        The model is a <b>Convolutional Neural Network (CNN)</b> built with <b>TensorFlow / Keras</b>.
        Input images are resized to <b>128 × 128 pixels</b> before being passed through the network.
        The trained model is saved in <code>.h5</code> format and loaded at startup with
        <code>@st.cache_resource</code> for fast repeated predictions without reloading weights.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DISEASE RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════
elif "Recognition" in page:

    st.markdown("""
    <div class="hero">
      <h1>🔬 Disease Recognition</h1>
      <p>Upload a leaf image to detect plant diseases instantly</p>
    </div>
    """, unsafe_allow_html=True)

    test_image = st.file_uploader(
        "UPLOAD LEAF IMAGE",
        type=["jpg", "jpeg", "png", "webp"],
        help="Clear, well-lit photos of individual leaves give the best results.",
    )

    if test_image is not None:
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown('<div class="section-label">🖼 Preview</div>', unsafe_allow_html=True)
            st.image(test_image, use_container_width=True)

        with right:
            st.markdown('<div class="section-label">🔬 Analysis</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-block" style="margin-bottom:1.2rem;">
                ✅ Image uploaded successfully.<br><br>
                Click <b>Predict Disease</b> to run the deep-learning model on your leaf image.
                Results and treatment advice will appear below.
            </div>
            """, unsafe_allow_html=True)

            predict_clicked = st.button("🔬  Predict Disease")

        if predict_clicked:
            with st.spinner("Running AI model…"):
                idx = model_prediction(test_image)
                name = CLASS_NAMES[idx]
                healthy = "Healthy" in name
                cls = "healthy" if healthy else "disease"
                icon = "✅" if healthy else "⚠️"
                badge = "Healthy Plant" if healthy else "Disease Detected"
                info = DISEASE_INFO.get(name, {})

            # ── Result badge ──────────────────────────────────────────────
            st.markdown(f"""
<div class="result-card {cls}">
<div class="r-icon">{icon}</div>
<div class="r-badge">{badge}</div>
<div class="r-name">{name}</div>
</div>
""", unsafe_allow_html=True)

            # ── Disease info panel ────────────────────────────────────────
            if info:
                symptoms_html = "".join(f"<li>{s}</li>" for s in info.get("symptoms", []))
                pill_cls = "" if healthy else "red"
                treatment_pills = "".join(
                    f'<span class="pill {pill_cls}">{t}</span>'
                    for t in info.get("treatment", [])
                )

                st.markdown(f"""
<div class="dinfo">
<h3>📋 About this Condition</h3>
<p>{info.get("description", "")}</p>

<h3 style="margin-top:1rem;">🔍 Key Symptoms</h3>
<ul>{symptoms_html}</ul>

<h3 style="margin-top:1rem;">💊 Treatment & Management</h3>
<div class="pill-row">{treatment_pills}</div>
</div>
""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="big">🌿</div>
            <div class="title">No image uploaded yet</div>
            <div>Use the uploader above to choose a plant leaf photo (JPG, PNG or WEBP).</div>
        </div>
        """, unsafe_allow_html=True)
