import pydicom
import os

# Change the patient ID and name 
def name_id(ds, new_patient_id, new_patient_name):
    ds.PatientID = new_patient_id
    ds.PatientName = new_patient_name
    print(f"Modified patient ID to: {new_patient_id}")
    print(f"Modified patient name to: {new_patient_name}")

    if 'OtherPatientNames' in ds:
        ds.OtherPatientNames = ''


# Only keep the age of patient, if the birth date is not available, remove it
def ageCalculation(ds):
    if 'PatientBirthDate' in ds and ds['PatientBirthDate'].value.strip():
        birth_date = ds.PatientBirthDate
        birthyear = birth_date[:4]
        birthyear = int(birthyear)
        # Get the year
        if 1900 < birthyear < 2100:
            if 'StudyDate' in ds:
                study_date = ds.StudyDate
                studyyear = study_date[:4]
                studyyear = int(studyyear)
                age = studyyear - birthyear
                print(f"Modified birth date to AGE: {age}")
                ds.PatientAge = str(age)
                ds.PatientBirthDate = ''
        else:
            ds.PatientBirthDate = ''
            ds.PatientAge = ''
            print("Birth date is empty or wrong, delete it.")
    else:
        ds.PatientAge = ''
        print("No birth date tag found, nothing changed.")


# Change the hospital name and remove the department, referring physician, acquisition date and time
def hospital(ds, new_hospital_name):
    if 'InstitutionName' in ds:
        ds.InstitutionName = new_hospital_name
        print(f"Modified hospital name to: {new_hospital_name}")
    else:
        print("No hospital name tag found, nothing changed.")

    if 'InstitutionalDepartmentName' in ds:
        ds.InstitutionalDepartmentName = ''
    
    if 'ReferringPhysicianName' in ds:
        ds.ReferringPhysicianName = ''
    
    if 'PhysiciansOfRecord' in ds:
        ds.PhysiciansOfRecord = ''
    
    if 'RequestingPhysician' in ds:
        ds.RequestingPhysician = ''


def remove_date_time(ds):
    # Tags for time and date
    date_time_tags = [
        'AcquisitionDate', 'AcquisitionTime',
        'ContentDate', 'ContentTime',
        'StudyDate', 'StudyTime', 'SeriesDate', 'SeriesTime',
        'ScheduledProcedureStepStartDate', 'ScheduledProcedureStepStartTime',
        'ScheduledProcedureStepEndDate', 'ScheduledProcedureStepEndTime',
        'PerformedProcedureStepStartDate', 'PerformedProcedureStepStartTime',
        'PerformedProcedureStepEndDate', 'PerformedProcedureStepEndTime',
        'InstitutionAddress', 'PatientAddress', 'PerformingPhysicianName',
        'OperatorsName', 'NameOfPhysiciansReadingStudy',
    ]
            
    # Setting tags in the specified range to a blank space if they exist
    for tag in date_time_tags:
        if tag in ds:
            ds.data_element(tag).value = ''

    print("All date and time removed.")
    

# New information
dicom_path = "/home/Craneal_CT/P"
save_path = "/home/Craneal_CT/new_P"
new_patient_id = 'P'
new_patient_name = 'P'
new_hospital_id = 'Hospital'

# Loop through all the files in the folder
for filename in os.listdir(dicom_path):
    if filename.lower().endswith('.dcm') or '.' not in filename:
        file_path = os.path.join(dicom_path, filename)
        ds = pydicom.dcmread(file_path)
        name_id(ds, new_patient_id, new_patient_name)
        ageCalculation(ds)
        hospital(ds, new_hospital_id)
        remove_date_time(ds)
        ds.save_as(os.path.join(save_path, filename))
