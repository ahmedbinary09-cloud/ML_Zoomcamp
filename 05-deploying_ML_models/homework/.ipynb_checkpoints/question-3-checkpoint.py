import pickle 

with open("pipeline.bin", "rb") as model_file:
    pipeline = pickle.load(model_file)

question_3 = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

answer = pipeline.predict_proba(question_3)[0, 1]
print(round(answer, 3))