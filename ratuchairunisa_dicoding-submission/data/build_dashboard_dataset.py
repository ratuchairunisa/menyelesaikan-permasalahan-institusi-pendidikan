"""
Script untuk membangun dataset final (readable) yang digunakan pada dashboard.
Mapping kode -> label diambil dari dokumentasi resmi dataset UCI:
"Predict students' dropout and academic success" (Realinho et al., 2021/2022).
"""
import pandas as pd

df = pd.read_csv('data/data_raw.csv', sep=';')

marital_map = {1:'Single',2:'Married',3:'Widower',4:'Divorced',5:'Facto Union',6:'Legally Separated'}
app_mode_map = {
    1:'1st phase - general contingent',2:'Ordinance No. 612/93',5:'1st phase - special contingent (Azores Island)',
    7:'Holders of other higher courses',10:'Ordinance No. 854-B/99',15:'International student (bachelor)',
    16:'1st phase - special contingent (Madeira Island)',17:'2nd phase - general contingent',
    18:'3rd phase - general contingent',26:'Ordinance No. 533-A/99, item b2) (Different Plan)',
    27:'Ordinance No. 533-A/99, item b3 (Other Institution)',39:'Over 23 years old',
    42:'Transfer',43:'Change of course',44:'Technological specialization diploma holders',
    51:'Change of institution/course',53:'Short cycle diploma holders',
    57:'Change of institution/course (International)'
}
course_map = {
    33:'Biofuel Production Technologies',171:'Animation and Multimedia Design',
    8014:'Social Service (evening attendance)',9003:'Agronomy',9070:'Communication Design',
    9085:'Veterinary Nursing',9119:'Informatics Engineering',9130:'Equinculture',
    9147:'Management',9238:'Social Service',9254:'Tourism',9500:'Nursing',
    9556:'Oral Hygiene',9670:'Advertising and Marketing Management',9773:'Journalism and Communication',
    9853:'Basic Education',9991:'Management (evening attendance)'
}
attendance_map = {1:'Daytime',0:'Evening'}
qualification_map = {
    1:'Secondary Education',2:"Higher Education - Bachelor's",3:'Higher Education - Degree',
    4:"Higher Education - Master's",5:'Higher Education - Doctorate',6:'Frequency of Higher Education',
    9:'12th Year - Not Completed',10:'11th Year - Not Completed',12:'Other - 11th Year',
    14:'10th Year',15:'10th Year - Not Completed',19:'Basic Education 3rd Cycle',
    38:'Basic Education 2nd Cycle',39:'Technological Specialization Course',
    40:'Higher Education - Degree (1st cycle)',42:'Professional Higher Technical Course',
    43:'Higher Education - Master (2nd cycle)'
}
gender_map = {1:'Male',0:'Female'}
yesno_map = {1:'Yes',0:'No'}

out = pd.DataFrame()
out['ID'] = range(1, len(df)+1)
out['Marital_Status'] = df['Marital_status'].map(marital_map).fillna('Other')
out['Application_Mode'] = df['Application_mode'].map(app_mode_map).fillna('Other')
out['Course'] = df['Course'].map(course_map).fillna('Other')
out['Attendance_Type'] = df['Daytime_evening_attendance'].map(attendance_map)
out['Previous_Qualification'] = df['Previous_qualification'].map(qualification_map).fillna('Other')
out['Admission_Grade'] = df['Admission_grade']
out['Displaced'] = df['Displaced'].map(yesno_map)
out['Special_Needs'] = df['Educational_special_needs'].map(yesno_map)
out['Debtor'] = df['Debtor'].map(yesno_map)
out['Tuition_Fees_Up_To_Date'] = df['Tuition_fees_up_to_date'].map(yesno_map)
out['Gender'] = df['Gender'].map(gender_map)
out['Scholarship_Holder'] = df['Scholarship_holder'].map(yesno_map)
out['Age_At_Enrollment'] = df['Age_at_enrollment']
out['International'] = df['International'].map(yesno_map)

out['Units_1st_Sem_Enrolled'] = df['Curricular_units_1st_sem_enrolled']
out['Units_1st_Sem_Approved'] = df['Curricular_units_1st_sem_approved']
out['Units_1st_Sem_Grade'] = df['Curricular_units_1st_sem_grade']
out['Units_2nd_Sem_Enrolled'] = df['Curricular_units_2nd_sem_enrolled']
out['Units_2nd_Sem_Approved'] = df['Curricular_units_2nd_sem_approved']
out['Units_2nd_Sem_Grade'] = df['Curricular_units_2nd_sem_grade']

# Derived, dashboard-friendly metrics
out['Approval_Rate_1st_Sem'] = (df['Curricular_units_1st_sem_approved'] /
                                  df['Curricular_units_1st_sem_enrolled'].replace(0, pd.NA)).fillna(0).round(3)
out['Approval_Rate_2nd_Sem'] = (df['Curricular_units_2nd_sem_approved'] /
                                  df['Curricular_units_2nd_sem_enrolled'].replace(0, pd.NA)).fillna(0).round(3)
out['Avg_Grade_Both_Sem'] = ((df['Curricular_units_1st_sem_grade'] + df['Curricular_units_2nd_sem_grade']) / 2).round(2)

out['Unemployment_Rate'] = df['Unemployment_rate']
out['Inflation_Rate'] = df['Inflation_rate']
out['GDP'] = df['GDP']
out['Status'] = df['Status']

out.to_csv('data/students_dashboard_final.csv', index=False)
print("Saved:", out.shape)
print(out.head(3).to_string())
print(out['Status'].value_counts())
