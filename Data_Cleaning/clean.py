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


# Only keep the year of the birth date
def birthday(ds):
    if 'PatientBirthDate' in ds:
        birth_date = ds.PatientBirthDate
        # Get the year
        if birth_date and len(birth_date) >= 4:
            new_birth_date = birth_date[:4]
            ds.PatientBirthDate = new_birth_date
            print(f"Modified birth date to: {new_birth_date}")
        else:
            print("Birth date is empty or too short, keeping original.")
    else:
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


def remove_date_time(ds):
    # Tags for time and date
    date_time_tags = [
        'AcquisitionDate', 'AcquisitionTime',
        'ContentDate', 'ContentTime',
        'StudyDate', 'StudyTime'
        'ScheduledProcedureStepStartDate', 'ScheduledProcedureStepStartTime',
        'ScheduledProcedureStepEndDate', 'ScheduledProcedureStepEndTime',
        'PerformedProcedureStepStartDate', 'PerformedProcedureStepStartTime',
        'PerformedProcedureStepEndDate', 'PerformedProcedureStepEndTime'
    ]
            
    # Setting tags in the specified range to a blank space if they exist
    for tag in date_time_tags:
        if tag in ds:
            ds.data_element(tag).value = ''

    print("All date and time removed.")
    

# New info.
dicom_path = "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/TAC Bueno/VCO"
save_path = "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/CleanCT"
new_patient_id = 'P01'
new_patient_name = 'P01'
new_hospital_id = 'A'

# Loop through all the files in the folder
for filename in os.listdir(dicom_path):
    if filename.lower().endswith('.dcm'):
        file_path = os.path.join(dicom_path, filename)
        ds = pydicom.dcmread(file_path)
        name_id(ds, new_patient_id, new_patient_name)
        birthday(ds)
        hospital(ds, new_hospital_id)
        remove_date_time(ds)
        ds.save_as(os.path.join(save_path, filename))