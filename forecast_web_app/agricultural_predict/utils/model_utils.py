import joblib




def save_model(model, model_url):
    joblib.dump(model, model_url)
    
    
    
def load_model(model_url):
    return joblib.load('regression_model.joblib')