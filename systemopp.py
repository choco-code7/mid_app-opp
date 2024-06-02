import pickle
import numpy as np
import streamlit as st
import os
from datetime import datetime
import matplotlib.pyplot as plt

class DiseaseModel:
    def __init__(self, model_file, normal_ranges, tooltips, deleted_features=None):
        self.model = self.load_model(model_file)
        self.normal_ranges = normal_ranges
        self.tooltips = tooltips
        self.deleted_features = deleted_features if deleted_features else []

    def load_model(self, model_file):
        with open(model_file, 'rb') as f:
            return pickle.load(f)

    def predict(self, user_input):
        prediction, confidence_percentage = self.model.predict_with_confidence(user_input)
        return prediction, confidence_percentage

    def display_normal_ranges(self):
        st.subheader("Normal Ranges")
        for feature, (min_val, max_val) in self.normal_ranges.items():
            if feature not in self.deleted_features:
                st.write(f"{feature}: {min_val} - {max_val}")

    def visualize_comparison(self, user_inputs):
        st.subheader("Comparison with Normal Ranges")
        user_input_index = 0
        for feature, (min_val, max_val) in self.normal_ranges.items():
            if feature in self.deleted_features:
                user_input_index += 1
                continue
            user_input = user_inputs[user_input_index]
            user_input_index += 1
            normal_range = max_val - min_val
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(0, user_input, color='#679EB6', label='User Input', height=0.4)
            ax.barh(0.5, normal_range, left=min_val, color='#B7CFCD', label='Normal Range', height=0.4)
            ax.set_yticks([0, 0.5])
            ax.set_yticklabels(['User Input', 'Normal Range'])
            ax.set_xlabel('Values')
            ax.set_title(f'{feature} Comparison')
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.text(user_input + 0.1, 0, f'{user_input:.2f}', va='center', ha='left')
            ax.text(min_val + normal_range / 2, 0.5, f'{min_val}-{max_val}', va='center', ha='center')
            st.pyplot(fig)

class DiabetesModel(DiseaseModel):
    pass

class BreastCancerModel(DiseaseModel):
    pass

class HeartDiseaseModel(DiseaseModel):
    pass

class ParkinsonsModel(DiseaseModel):
    pass

def load_models():
    model_dir = '/Users/ahmedmoataz/Documents/project/appopp/saved_models'
    model_files = {
        "diabetes": os.path.join(model_dir, "diabetes_model.sav"),
        "breast cancer": os.path.join(model_dir, "breast_cancer_model.sav"),
        "heart disease": os.path.join(model_dir, "heart_model.sav"),
        "parkinson's disease": os.path.join(model_dir, "parkinsons_model.sav"),
    }

    normal_ranges = {
        "diabetes": {'Pregnancies': (0, 10), 'Glucose': (70, 140), 'BloodPressure': (60, 110), 'SkinThickness': (10, 50), 'Insulin': (0, 300), 'BMI': (18.5, 24.9), 'DiabetesPedigreeFunction': (0, 2.5), 'Age': (0, 100)},
        "breast cancer": {'radius_mean': (0, 50), 'texture_mean': (0, 50), 'perimeter_mean': (0, 250), 'area_mean': (0, 2500), 'smoothness_mean': (0, 0.3), 'compactness_mean': (0, 0.5), 'concavity_mean': (0, 0.5), 'concave_points_mean': (0, 0.3), 'symmetry_mean': (0, 1), 'fractal_dimension_mean': (0, 0.5), 'radius_se': (0, 5), 'texture_se': (0, 5), 'perimeter_se': (0, 50), 'area_se': (0, 500), 'smoothness_se': (0, 0.1), 'compactness_se': (0, 0.3), 'concavity_se': (0, 0.3), 'concave_points_se': (0, 0.1), 'symmetry_se': (0, 0.5), 'fractal_dimension_se': (0, 0.1), 'radius_worst': (0, 50), 'texture_worst': (0, 50), 'perimeter_worst': (0, 250), 'area_worst': (0, 2500), 'smoothness_worst': (0, 0.3), 'compactness_worst': (0, 1), 'concavity_worst': (0, 1), 'concave_points_worst': (0, 0.5), 'symmetry_worst': (0, 1), 'fractal_dimension_worst': (0, 1)},
        "heart disease": {'age': (0, 100), 'sex': (0, 1), 'cp': (0, 3), 'trestbps': (80, 200), 'chol': (100, 600), 'fbs': (0, 1), 'restecg': (0, 2), 'thalach': (60, 220), 'exang': (0, 1), 'oldpeak': (0, 10), 'slope': (0, 2), 'ca': (0, 4), 'thal': (0, 3)},
        "parkinson's disease": {'fo': (50, 300), 'fhi': (50, 300), 'flo': (50, 300), 'Jitter_percent': (0, 1), 'Jitter_Abs': (0, 1), 'RAP': (0, 1), 'PPQ': (0, 1), 'DDP': (0, 1), 'Shimmer': (0, 1), 'Shimmer_dB': (0, 1), 'APQ3': (0, 1), 'APQ5': (0, 1), 'APQ': (0, 1), 'DDA': (0, 1), 'NHR': (0, 1), 'HNR': (0, 50), 'RPDE': (0, 1), 'DFA': (0, 1), 'spread1': (0, 1), 'spread2': (0, 1), 'D2': (0, 1), 'PPE': (0, 1)},
    }

    tooltips = {
        "diabetes": [
            'Number of pregnancies',
            'Glucose level (in mg/dL)',
            'Blood pressure (mm Hg)',
            'Skin thickness (in mm)',
            'Insulin level (in mU/L)',
            'BMI (Body Mass Index)',
            'Diabetes pedigree function (value)',
            'Age (in years)'
        ],
        "breast cancer": [
            'Mean radius (mean of distances from center to points on the perimeter)',
            'Mean texture (standard deviation of gray-scale values)',
            'Mean perimeter',
            'Mean area',
            'Mean smoothness (local variation in radius lengths)',
            'Mean compactness (perimeter^2 / area - 1.0)',
            'Mean concavity (severity of concave portions of the contour)',
            'Mean concave points (number of concave portions of the contour)',
            'Mean symmetry',
            'Mean fractal dimension ("coastline approximation" - 1)',
            'Standard error of radius',
            'Standard error of texture',
            'Standard error of perimeter',
            'Standard error of area',
            'Standard error of smoothness',
            'Standard error of compactness',
            'Standard error of concavity',
            'Standard error of concave points',
            'Standard error of symmetry',
            'Standard error of fractal dimension',
            'Worst radius (largest tumor dimension)',
            'Worst texture (standard deviation of gray-scale values)',
            'Worst perimeter',
            'Worst area',
            'Worst smoothness',
            'Worst compactness',
            'Worst concavity',
            'Worst concave points',
            'Worst symmetry',
            'Worst fractal dimension'
        ],
        "heart disease": [
            'Age (in years)',
            'Sex (1 for male, 0 for female)',
            'Chest pain type (0-3)',
            'Resting blood pressure (mm Hg)',
            'Serum cholesterol (mg/dL)',
            'Fasting blood sugar (> 120 mg/dL, 1 = true; 0 = false)',
            'Resting electrocardiographic results (0-2)',
            'Maximum heart rate achieved',
            'Exercise induced angina (1 = yes; 0 = no)',
            'ST depression induced by exercise relative to rest',
            'Slope of the peak exercise ST segment (0-2)',
            'Number of major vessels colored by fluoroscopy',
            'Thalium stress test result (3 = normal; 6 = fixed defect; 7 = reversable defect)'
        ],
        "parkinson's disease": [
            'MDVP:Fo(Hz) (fundamental frequency)',
            'MDVP:Fhi(Hz) (highest frequency)',
            'MDVP:Flo(Hz) (lowest frequency)',
            'MDVP:Jitter(%) (measure of frequency variation)',
            'MDVP:Jitter(Abs) (absolute value of jitter)',
            'MDVP:RAP (relative amplitude perturbation)',
            'MDVP:PPQ (five-point period perturbation quotient)',
            'Jitter:DDP (average absolute difference of differences between jitter cycles)',
            'MDVP:Shimmer (local amplitude variation)',
            'MDVP:Shimmer(dB) (logarithmic scaled version of MDVP:Shimmer)',
            'Shimmer:APQ3 (three-point amplitude perturbation quotient)',
            'Shimmer:APQ5 (five-point amplitude perturbation quotient)',
            'MDVP:APQ (average amplitude perturbation quotient)',
            'Shimmer:DDA (average absolute differences between consecutive differences)',
            'NHR (noise-to-harmonics ratio)',
            'HNR (harmonics-to-noise ratio)',
            'RPDE (recurrence period density entropy measure)',
            'DFA (signal fractal scaling exponent)',
            'spread1 (nonlinear measure of fundamental frequency variation)',
            'spread2 (nonlinear measure of fundamental frequency variation)',
            'D2 (correlation dimension)',
            'PPE (pitch period entropy)'
        ]
    }

    deleted_features = ['Sex', 'Age', 'Blood_Pressure_Abnormality', 'Smoking', 'Chronic_kidney_disease', 'Physical_activity', 'age', 'sex', 'exang', 'cp', 'fbs', 'Gender']

    models = {
        "diabetes": DiabetesModel(model_files["diabetes"], normal_ranges["diabetes"], tooltips["diabetes"], deleted_features),
        "breast cancer": BreastCancerModel(model_files["breast cancer"], normal_ranges["breast cancer"], tooltips["breast cancer"], deleted_features),
        "heart disease": HeartDiseaseModel(model_files["heart disease"], normal_ranges["heart disease"], tooltips["heart disease"], deleted_features),
        "parkinson's disease": ParkinsonsModel(model_files["parkinson's disease"], normal_ranges["parkinson's disease"], tooltips["parkinson's disease"], deleted_features),
    }

    return models

models = load_models()

def main():
    st.sidebar.title("Disease Prediction Using ML")
    st.sidebar.markdown("---")
    
    select = st.sidebar.selectbox("Choose a Disease", ['No Selection'] + [disease.title() for disease in models.keys()])
    
    if select != 'No Selection':
        st.title(f'{select} Prediction')
        model = models[select.lower()]
        inputs = model.tooltips
        user_inputs = [st.number_input(input_name, step=0.1, format="%g", value=None, help=tooltip) for input_name, tooltip in zip(model.normal_ranges.keys(), inputs)]
        
        if st.button('Predict'):
            user_input = np.array(user_inputs).reshape(1, -1)
            
            valid_input = True
            error_message = ""
            negative_allowed_features = ["spread1"]
            
            for i, input_value in enumerate(user_input[0]):
                if input_value is None:
                    error_message += f"Empty input for {list(model.normal_ranges.keys())[i]}. Please provide a value.\n"
                    valid_input = False
                elif input_value < 0 and list(model.normal_ranges.keys())[i] not in negative_allowed_features:
                    error_message += f"Invalid input for {list(model.normal_ranges.keys())[i]}. Value must be non-negative.\n"
                    valid_input = False
            
            if not valid_input:
                st.error(error_message)
            else:
                prediction, confidence_percentage = model.predict(user_input)
                if prediction == 1:
                    st.warning(f'The prediction result is positive. There is a {confidence_percentage[0]:.0f}% confidence that the disease is present.')
                else:
                    st.success(f'The prediction result is negative. There is a {confidence_percentage[0]:.0f}% confidence that the disease is absent.')
                
                model.display_normal_ranges()
                model.visualize_comparison(user_inputs)
    else:
        st.title('Welcome to the Disease Prediction System')
        st.markdown("---")
        instructions = [
            "Choose a disease from the sidebar to predict its occurrence.",
            "Fill in your relevant medical data into the provided fields.",
            "Hover over the tooltip icon <i class='fas fa-question-circle'></i> for additional information on each input field.",
            "Ensure all inputs are numeric; text input is not allowed.",
            "Click on the 'Predict' button to obtain your prediction result.",
            "Review the comparison between your input data and the normal range to assess your results."
        ]
        
        st.markdown("#### Instructions:")
        st.markdown("\n".join([f"- {instruction}" for instruction in instructions]), unsafe_allow_html=True)
    
    current_year = datetime.now().year
    footer = f"""
    ---
    © {current_year} Disease Prediction System | Developed by Ahmed Mohamed
    """
    st.markdown(footer)

if __name__ == "__main__":
    main()
