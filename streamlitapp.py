import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from geopy.geocoders import Nominatim
from scipy.special import inv_boxcox
import scipy.stats as sps
from sklearn.metrics import mean_absolute_error

# ---------------- Config ----------------
BEST_LAMBDA = -0.03  # set this to your chosen Box–Cox lambda

st.set_page_config(
    page_title="Buenos Aires Apartment Price Dashboard",
    layout="wide",
)

# ---------------- Load artifacts ----------------
@st.cache_resource
def load_model_and_scaler():
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("standard_scaler.pkl", "rb") as f:
        std_scaler = pickle.load(f)
    return model, std_scaler

model, std_scaler = load_model_and_scaler()

@st.cache_data
def load_data():
    # df_full_buenos_aires.pkl must be created from your wrangled df_full
    with open("df_full_buenos_aires.pkl", "rb") as f:
        df_full = pickle.load(f)
    return df_full

df_full = load_data()

# ---------------- Diagnostics data (drop NaNs) ----------------
diag_df = df_full[["latitude", "longitude", "surface_covered_in_m2", "price_aprox_usd"]].dropna()

X_all = diag_df[["latitude", "longitude", "surface_covered_in_m2"]].copy()
y_all = diag_df["price_aprox_usd"].copy()

X_bc = X_all.copy()
X_bc["surface_covered_in_m2"] = sps.boxcox(X_bc["surface_covered_in_m2"], BEST_LAMBDA)
X_scaled = pd.DataFrame(
    std_scaler.transform(X_bc),
    columns=X_bc.columns,
    index=X_bc.index,
)
y_bc = sps.boxcox(y_all, BEST_LAMBDA)
y_pred_bc = model.predict(X_scaled)
resid_bc = y_bc - y_pred_bc

# ---------------- Geocoder ----------------
geolocator = Nominatim(user_agent="ba_price_app")

def get_address(latitude, longitude):
    try:
        location = geolocator.reverse(
            (latitude, longitude),
            exactly_one=True,
            timeout=10,
            addressdetails=True,
            language="en",
            zoom=16,
        )
        return location.address
    except Exception:
        return "Not Found!"

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")

muni_options = sorted(df_full["municipality"].unique())
selected_munis = st.sidebar.multiselect(
    "Municipality",
    options=muni_options,
    default=muni_options,
)

min_price = float(df_full["price_aprox_usd"].min())
max_price = float(df_full["price_aprox_usd"].max())
price_range = st.sidebar.slider(
    "Price range (USD)",
    min_value=int(min_price),
    max_value=int(max_price),
    value=(int(min_price), int(max_price)),
    step=1000,
)

min_cov = float(df_full["surface_covered_in_m2"].min())
max_cov = float(df_full["surface_covered_in_m2"].max())
covered_range = st.sidebar.slider(
    "Covered surface range (m²)",
    min_value=int(min_cov),
    max_value=int(max_cov),
    value=(int(min_cov), int(max_cov)),
    step=5,
)

mask = (
    df_full["municipality"].isin(selected_munis)
    & df_full["price_aprox_usd"].between(price_range[0], price_range[1])
    & df_full["surface_covered_in_m2"].between(covered_range[0], covered_range[1])
)
df = df_full[mask].copy()

st.markdown(
    "<h1 style='text-align:center; color:#FD8916;'>Buenos Aires Apartment Price Dashboard</h1>",
    unsafe_allow_html=True,
)

# ---------------- Tabs ----------------
tab_predict, tab_eda, tab_model = st.tabs(["🔮 Predict", "📊 EDA", "📉 Model"])

# ===== Tab 1: Prediction =====
with tab_predict:
    col_left, col_right = st.columns([0.45, 0.55])

    with col_left:
        st.markdown(
            "<div style='border:2px solid #17616E; border-radius:10px; padding:10px;'>",
            unsafe_allow_html=True,
        )

        covered_surface = st.number_input(
            "Covered surface in m²",
            min_value=5.0,
            max_value=500.0,
            step=1.0,
            value=None,
            key="covered_surface_input",
        )

        lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            step=0.0001,
            format="%.6f",
            value=None,
            key="latitude_input",
        )

        lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            step=0.0001,
            format="%.6f",
            value=None,
            key="longitude_input",
        )

        predict_button = st.button(
            "Predict Price",
            type="primary",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if predict_button and all(v is not None for v in [covered_surface, lat, lon]):
            address = get_address(lat, lon)
            try:
                city = [el.strip() for el in address.split(",")][-3]
            except Exception:
                st.error("So you want to buy an apartment in the sea... Very funny 😄")
                st.stop()

            if city != "Autonomous City of Buenos Aires":
                st.error(
                    "❌ The city is not in the Autonomous City of Buenos Aires! "
                    "Please select another location."
                )
                st.stop()

            features = pd.DataFrame(
                [[lat, lon, covered_surface]],
                columns=["latitude", "longitude", "surface_covered_in_m2"],
            )

            features["surface_covered_in_m2"] = sps.boxcox(
                features["surface_covered_in_m2"], BEST_LAMBDA
            )

            features_scaled = std_scaler.transform(features)
            features_df = pd.DataFrame(features_scaled, columns=features.columns)

            pred_trans = model.predict(features_df)
            pred_price = inv_boxcox(pred_trans, BEST_LAMBDA)[0]

            with st.container():
                st.markdown(
                    "<div style='border:2px solid #17616E; border-radius:10px; "
                    "padding:10px; margin-top:10px;'>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='font-weight:bold; font-family:Calibri;'>Predicted Price: "
                    f"<span style='color:#FD8916;'>${pred_price:,.0f}</span></p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<p style='font-weight:bold; font-family:Calibri;'>Apartment Address:</p>",
                    unsafe_allow_html=True,
                )
                st.write(address)

                try:
                    parts = [el.strip().lower() for el in address.split(",")]
                    info_mun = df_full[
                        df_full["municipality"].str.lower().str.contains(parts[1])
                        | df_full["municipality"].str.lower().str.contains(parts[2])
                        | df_full["municipality"].str.lower().str.contains(parts[3])
                    ]
                except Exception:
                    st.info(
                        "🔱 Poseidon has put up a ‘No Humans Allowed’ sign for the sea apartments."
                    )
                    info_mun = pd.DataFrame()

                if info_mun.shape[0] == 0:
                    st.info("🤷🏻‍♂️ It seems that we do not have any information about this location.")
                else:
                    municipality_name = info_mun["municipality"].unique()[0]
                    st.markdown(
                        f"<p style='font-weight:bold; font-family:Calibri;'>Municipality Statistics: "
                        f"<span style='color:#FD8916;'>{municipality_name}</span></p>",
                        unsafe_allow_html=True,
                    )

                    stats = pd.DataFrame(info_mun["price_aprox_usd"].describe())[1:]
                    stats["Measure"] = [
                        "Average Price (USD)",
                        "Price Standard Deviation (USD)",
                        "Minimum Price (USD)",
                        "25th Percentile Price (USD)",
                        "Median Price (USD)",
                        "75th Percentile Price (USD)",
                        "Maximum Price (USD)",
                    ]
                    stats = stats[["Measure", "price_aprox_usd"]].rename(
                        columns={"price_aprox_usd": "Value"}
                    )
                    stats["Value"] = stats["Value"].apply(
                        lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x
                    )

                    st.dataframe(
                        stats.reset_index(drop=True),
                        use_container_width=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown(
            "<div style='border-radius:10px; padding:10px; background-color:#173b61;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-weight:bold; font-family:Calibri; color:white;'>"
            "Sample of listings in Buenos Aires (filtered)</p>",
            unsafe_allow_html=True,
        )

        if df.shape[0] > 0:
            fig_map = px.scatter_mapbox(
                df.sample(min(1000, len(df))),
                lat="latitude",
                lon="longitude",
                hover_data=["price_aprox_usd", "municipality"],
                color="price_aprox_usd",
                color_continuous_scale="Viridis",
                zoom=11,
                height=600,
            )
            fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No data for current filters.")

        st.markdown("</div>", unsafe_allow_html=True)

# ===== Tab 2: EDA =====
with tab_eda:
    if df.shape[0] == 0:
        st.info("No data for current filters.")
    else:
        st.subheader("Average price by municipality")

        avg_price = (
            df.groupby("municipality")["price_aprox_usd"]
            .mean()
            .sort_values(ascending=True)
            .reset_index()
        )
        fig_bar = px.bar(
            avg_price,
            x="price_aprox_usd",
            y="municipality",
            orientation="h",
            labels={"price_aprox_usd": "Average price (USD)", "municipality": ""},
            color="price_aprox_usd",
            color_continuous_scale="Tealgrn",
            height=600,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Price vs. latitude and longitude (3D)")
        fig_3d = px.scatter_3d(
            df,
            x="latitude",
            y="longitude",
            z="price_aprox_usd",
            color="price_aprox_usd",
            labels={
                "longitude": "Longitude",
                "latitude": "Latitude",
                "price_aprox_usd": "Price in USD",
            },
            template="plotly_white",
            height=600,
        )
        fig_3d.update_traces(
            marker=dict(size=4, line=dict(width=1, color="white")),
            selector=dict(mode="markers"),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.subheader("Price vs. surface (total & covered)")
        col1, col2 = st.columns(2)
        with col1:
            if "surface_total_in_m2" in df.columns:
                fig_total = px.scatter(
                    df,
                    x="surface_total_in_m2",
                    y="price_aprox_usd",
                    color="price_aprox_usd",
                    title="Price vs. total surface (m²)",
                    labels={
                        "surface_total_in_m2": "Total surface (m²)",
                        "price_aprox_usd": "Price (USD)",
                    },
                    template="plotly_white",
                    height=500,
                )
                fig_total.update_traces(
                    marker=dict(size=6, line=dict(width=1, color="white")),
                    selector=dict(mode="markers"),
                )
                st.plotly_chart(fig_total, use_container_width=True)
        with col2:
            fig_cov = px.scatter(
                df,
                x="surface_covered_in_m2",
                y="price_aprox_usd",
                color="price_aprox_usd",
                title="Price vs. covered surface (m²)",
                labels={
                    "surface_covered_in_m2": "Covered surface (m²)",
                    "price_aprox_usd": "Price (USD)",
                },
                template="plotly_white",
                height=500,
            )
            fig_cov.update_traces(
                marker=dict(size=6, line=dict(width=1, color="white")),
                selector=dict(mode="markers"),
            )
            st.plotly_chart(fig_cov, use_container_width=True)

# ===== Tab 3: Model diagnostics =====
with tab_model:
    st.subheader("Gradient Boosting residuals vs. transformed target")

    fig_resid = px.scatter(
        x=y_bc,
        y=resid_bc,
        labels={"x": "Transformed price (Box–Cox)", "y": "Residuals (GB)"},
        template="plotly_white",
        height=500,
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_resid, use_container_width=True)

    mae_full = mean_absolute_error(y_all, inv_boxcox(y_pred_bc, BEST_LAMBDA))
    st.write(f"Diagnostics MAE on complete cases: ${mae_full:,.0f}")

st.markdown(
    "<p style='text-align:center; font-family:Calibri; font-size:12px;'>"
    "Built with Streamlit, scikit-learn, and GeoPy. Data: Buenos Aires real estate.</p>",
    unsafe_allow_html=True,
)
