# Buenos-Aires-Real-Estate-Price-Prediction
Interactive dashboard to predict apartment prices in Ciudad Autónoma de Buenos Aires (CABA) using a Gradient Boosting Regressor trained on Properati real estate data (2016).

## Features
•	Live Price Prediction: Enter covered surface, latitude/longitude → Get instant USD price prediction
•	Interactive EDA: Filterable plots (price by municipality, 3D spatial, price vs. surface)
•	Model Diagnostics: Residuals analysis and performance metrics
•	CABA Map: Visualize listings colored by price
•	Municipality Stats: Price distribution for clicked/predicted locations

## Clone repo
git clone https://github.com/Bartho-A/Buenos-Aires-Real-Estate-Price-Prediction.git
cd Buenos-Aires-Real-Estate-Price-Prediction

## Create environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
## venv\Scripts\activate  # Windows

## Install dependencies
pip install -r requirements.txt

## Run app
streamlit run streamlitapp.py

├── streamlitapp.py          # Main Streamlit dashboard
├── final_model.pkl          # Trained Gradient Boosting model
├── standard_scaler.pkl      # Fitted StandardScaler
├── df_full_buenos_aires.pkl # Cleaned CABA apartment data
├── requirements.txt         # Dependencies
└── README.md               # You're reading it!

## Methodology
1.	Data: Properati Buenos Aires listings (Sep–Oct 2016)
2.	Cleaning: CABA apartments only, price < $400K, outlier removal
3.	Features:  latitude ,  longitude ,  surface_covered_in_m2 
4.	Preprocessing: Box–Cox (λ=-0.03) + Standardization
5.	Model: Gradient Boosting Regressor (n_estimators=1000)
6.	Validation: Filtered influential points (Cook’s D > 0.0003, |resid| > 3)

## Key Insights
Northern/Eastern CABA = Premium prices (Recoleta, Palermo, Puerto Madero)
Price ∝ Surface area (moderate positive correlation)
Spatial gradients captured via lat/lon coordinates

## Production Notes
•	Geocoding: Nominatim (OpenStreetMap) with CABA validation
•	Caching:  @st.cache_resource/data  for model/data loading
•	Error Handling: Invalid locations rejected gracefully
•	Responsive: Works on desktop/mobile

## Deployment
Deployed to Posit Cloud via GitHub integration:
1.	Connect repo → Python 3.11
2.	Main file:  streamlitapp.py 
3.	Install  requirements.txt 

## Contributing
1.	Fork the repo
2.	Create feature branch ( git checkout -b feature/eda-enhancements )
3.	Commit changes ( git commit -m 'Add municipality price heatmap' )
4.	Push ( git push origin feature/eda-enhancements )
5.	Open Pull Request

## License
MIT License - see LICENSE © 2025 Bartho Aobe
