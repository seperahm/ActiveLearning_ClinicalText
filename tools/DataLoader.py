import pandas as pd 

class DataLoader:
    def __init__(self,
                 # result_csv_file_path = "/home/saveuser/S/projects/rawan2_project/Cleaned_dataset/Race_Dataset_CLEANED.csv"
                 result_csv_file_path = "Race_Dataset_CLEANED.csv"
                ):
        self.result_csv_file_path = result_csv_file_path
        
    def load_data(self):
        df = pd.read_csv(self.result_csv_file_path)
        infor_columns = df.columns[:15]
        # retrieve columns that belong to different category
        place_of_birth_related = [column for column in infor_columns if 'POB' in column]
        race_related = [column for column in infor_columns if 'Race' in column]
        citizenship_related = [column for column in infor_columns if 'Citizenship' in column]
        # append patient id and text
        patient_id_column = df['patient_id']
        text_column = df['text']
        # with characteristic Status and if it's Assumed
        place_of_birth_full = pd.concat([patient_id_column,text_column,df[place_of_birth_related]],axis=1)
        race_full = pd.concat([patient_id_column,text_column,df[race_related]],axis=1)
        citizenship_full = pd.concat([patient_id_column,text_column,df[citizenship_related]],axis=1)
        # only with characteristic labels
        race_label_column = df['Race_Label']
        pob_label_column = df['POB_Label']
        citizenship_label_column = df['Citizenship_Label']
        place_of_birth = pd.concat([patient_id_column,text_column,pob_label_column],axis=1)
        race = pd.concat([patient_id_column,text_column,race_label_column],axis=1)
        citizenship = pd.concat([patient_id_column,text_column,citizenship_label_column],axis=1)
        data = {"place_of_birth_full":place_of_birth_full,
                "place_of_birth": place_of_birth,
                "race_full": race_full,
                "race": race,
                "citizenship_full": citizenship_full,
                "citizenship": citizenship}
        return data