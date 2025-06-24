from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle
import xgboost as xgb
import os
import logging
from werkzeug.exceptions import BadRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Global variables for model and encoders
model = None
encoders = None
caste_encoder = None
branch_encoder = None
college_encoder = None

def load_model_and_encoders():
    """Load ML model and encoders with error handling"""
    global model, encoders, caste_encoder, branch_encoder, college_encoder
    
    try:
        # Load XGBoost model
        model_path = "model/model_1.json"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Load encoders
        encoders_path = "model/label_encoders.pkl"
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders file not found: {encoders_path}")
        
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)
        
        caste_encoder = encoders["caste"]
        branch_encoder = encoders["branch"]
        college_encoder = encoders["college"]
        logger.info("Encoders loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model/encoders: {str(e)}")
        raise

def validate_input(caste, percentile, rank, branch):
    """Validate user input"""
    errors = []
    
    # Validate caste
    valid_castes = ['General', 'SC', 'ST', 'GVJS', 'GNT1S', 'GNT2S', 'GNT3S', 'OBC',
       'GSEBCS', 'LVJS', 'LNT2S', 'LSEBCS', 'PWDOPENS', 'PWDOBCS',
       'DEFOPENS', 'DEFOBCS', 'TFWS', 'PWDROBC', 'DEFRSEBC', 'EWS',
       'LNT1S', 'LNT3S', 'PWDRSCS', 'DEFROBCS', 'ORPHAN', 'DEFRNT1S',
       'GNT2H', 'GSEBCH', 'LOPENH', 'LOBCH', 'LSEBCH', 'GSCO', 'GVJO',
       'LOPENO', 'GVJH', 'GOPENO', 'GOBCO', 'GSEBCO', 'LOBCO', 'LSCH',
       'GNT3H', 'LSTH', 'LVJH', 'LNT1H', 'PWDOPENH', 'GNT1O', 'GNT2O',
       'GNT3O', 'LSTO', 'GNT1H', 'LNT3H', 'PWDOBCH', 'GSTO', 'LSCO',
       'LVJO', 'LNT2O', 'LNT2H', 'LSEBCO', 'LNT1O', 'LNT3O', 'DEFSCS',
       'PWDRSTS', 'MI', 'PWDRSEBC', 'PWDRNT2S', 'DEFRSCS', 'DEFRNT2S',
       'PWDSCS', 'DEFSEBCS', 'DEFRVJS', 'PWDSCH', 'PWDSEBCS', 'PWDRSCH',
       'PWDRSTH', 'DEFRNT3S', 'PWDRVJS', 'PWDRNT3S', 'PWDRNT1S', 'DEFSTS']
    if caste not in valid_castes:
        errors.append("Invalid caste selection")
    
    # Validate percentile
    try:
        percentile = float(percentile)
        if not (0 <= percentile <= 100):
            errors.append("Percentile must be between 0 and 100")
    except ValueError:
        errors.append("Invalid percentile format")
    
    # Validate rank
    try:
        rank = int(rank)
        if rank <= 0:
            errors.append("Rank must be a positive integer")
    except ValueError:
        errors.append("Invalid rank format")
    
    # Validate branch
    valid_branches = ['Civil Engineering', 'Computer Science and Engineering', 'Information Technology', 'Electrical Engineering', 'Electronics and Telecommunication Engg', 'Instrumentation Engineering', 'Mechanical Engineering', 'Food Technology', 'Oil and Paints Technology', 'Paper and Pulp Technology', 'Petro Chemical Engineering', 'Computer Engineering', 'Electrical Engg[Electronics and Power]', 'Artificial Intelligence (AI) and Data Science', 'Computer Science and Engineering (IoT)', 'Computer Science and Engineering(Artificial Intelligence and Machine Learning)', 'Artificial Intelligence and Data Science', 'Chemical Engineering', 'Textile Engineering / Technology', 'Computer Science and Engineering(Data Science)', 'Production Engineering', 'Textile Technology', 'Pharmaceutical and Fine Chemical Technology', 'Artificial Intelligence and Machine Learning', 'Electronics and Computer Engineering', 'VLSI', '5G', 'Agricultural Engineering', 'Computer Science and Design', 'Plastic and Polymer Engineering', 'Mechatronics Engineering', 'Mechanical & Automation Engineering', 'Automation and Robotics', 'Architectural Assistantship', 'Electronics Engineering ( VLSI Design and Technology)', 'Electrical and Electronics Engineering', 'Electronics and Computer Science', 'Safety and Fire Engineering', 'Electronics and Communication(Advanced Communication Technology)', 'Artificial Intelligence', 'Computer Science', 'Electronics Engineering', 'Production Engineering[Sandwich]', 'Dyestuff Technology', 'Oil,Oleochemicals and Surfactants Technology', 'Pharmaceuticals Chemistry and Technology', 'Fibres and Textile Processing Technology', 'Polymer Engineering and Technology', 'Food Engineering and Technology', 'Surface Coating Technology', 'Food Technology And Management', 'Civil and infrastructure Engineering', 'Bio Medical Engineering', 'Computer Science and Engineering (Internet of Things and Cyber Security Including Block Chain', 'Data Science', 'Cyber Security', 'Electronics and Communication (Advanced Communication Technology)', 'Automobile Engineering', 'Internet of Things (IoT)', 'Mechanical and Mechatronics Engineering (Additive Manufacturing)', 'Computer Science and Engineering(Cyber Security)', 'Mechanical Engineering Automobile', 'Electrical and Computer Engineering', 'Civil Engineering and Planning', 'Electronics and Biomedical Engineering', 'Computer Science and Information Technology', 'Data Engineering', 'Computer Technology', 'Electronics and Communication Engineering', 'Computer Science and Engineering (Cyber Security)', 'Computer Science and Engineering (Artificial Intelligence)', 'Aeronautical Engineering', 'Bio Technology', 'Industrial IoT', 
                      'Robotics and Artificial Intelligence', 'Mining Engineering', 'Computer Science and Business Systems', 'Fire Engineering', 'Plastic Technology', 'Oil Fats and Waxes Technology', 'Paints Technology', 'Instrumentation and Control Engineering', 'Robotics and Automation', 'Structural Engineering', 'Civil and Environmental Engineering', 'Computer Science and Technology', 'Computer Engineering (Software Engineering)', 'Technical Textiles', 'Fashion Technology', 'Man Made Textile Technology', 'Textile Chemistry', 'Computer Science and Engineering (Artificial Intelligence and Data Science)', 'Printing and Packing Technology', 'Mechanical Engineering[Sandwich]', 'Electrical, Electronics and Power', 'Oil Technology', 'Manufacturing Science and Engineering', 'Metallurgy and Material Technology']
    if branch not in valid_branches:
        errors.append("Invalid branch selection")
    
    return errors

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get form data
            caste = request.form.get("caste", "").strip()
            percentile = request.form.get("percentile", "").strip()
            rank = request.form.get("rank", "").strip()
            branch = request.form.get("branch", "").strip()
            
            # Validate input
            errors = validate_input(caste, percentile, rank, branch)
            if errors:
                for error in errors:
                    flash(error, 'error')
                return render_template("index.html")
            
            # Convert to proper types
            percentile = float(percentile)
            rank = int(rank)
            
            # Check if encoders can handle the input
            try:
                caste_encoded = caste_encoder.transform([caste])[0]
                branch_encoded = branch_encoder.transform([branch])[0]
            except ValueError as e:
                flash(f"Input encoding error: {str(e)}", 'error')
                return render_template("index.html")
            
            # Create input vector
            input_vector = np.array([[
                caste_encoded,
                percentile,
                rank,
                branch_encoded
            ]])
            
            # Make prediction
            proba = model.predict_proba(input_vector)[0]
            college_probs = {
                college_encoder.inverse_transform([i])[0]: p 
                for i, p in enumerate(proba)
            }
            
            # Get top 20 colleges with probabilities
            top_colleges = sorted(
                college_probs.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]
            
            # Create list of college names with their probabilities
            top_20_colleges = []
            college_chances = []
            
            for name, prob in top_colleges:
                top_20_colleges.append(name)
                college_chances.append(round(prob * 100, 2))
            
            # Pass to result page (fixed typo: banch -> branch)
            return render_template(
                "result.html", 
                colleges=top_20_colleges,
                chances=college_chances,
                branch=branch,
                user_data={
                    'caste': caste,
                    'percentile': percentile,
                    'rank': rank
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            flash("An error occurred during prediction. Please try again.", 'error')
            return render_template("index.html")
    
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions (optional)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        caste = data.get('caste')
        percentile = data.get('percentile')
        rank = data.get('rank')
        branch = data.get('branch')
        
        errors = validate_input(caste, percentile, rank, branch)
        if errors:
            return jsonify({'errors': errors}), 400
        
        # Make prediction (same logic as above)
        input_vector = np.array([[
            caste_encoder.transform([caste])[0],
            float(percentile),
            int(rank),
            branch_encoder.transform([branch])[0]
        ]])
        
        proba = model.predict_proba(input_vector)[0]
        college_probs = {
            college_encoder.inverse_transform([i])[0]: float(p) 
            for i, p in enumerate(proba)
        }
        
        top_colleges = sorted(
            college_probs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
        
        return jsonify({
            'success': True,
            'colleges': [{'name': name, 'probability': prob} for name, prob in top_colleges]
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == "__main__":
    try:
        load_model_and_encoders()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error: {str(e)}")
        print("Please ensure model files exist in the 'model' directory")