import matplotlib.pyplot as plt
import sys
import csv
import numpy as np
import pandas as pd

def imported_file(file_name):
    try:
        data, columns, column_mean=[], [], []
        with open(file_name, 'r', encoding='utf-8') as open_file:
            csv_reader = csv.reader(open_file, quotechar='"', delimiter=',',
                                quoting=csv.QUOTE_ALL, skipinitialspace=True)

            data = list(csv_reader)
            selected_columns=['popularity', 'danceability', 'energy', 'key', 'loudness', 'acousticness', 'liveness', 'duration_ms',
                 'streaming_count']
            for col in range(len(data[0])):
                columns.append([])
                column_mean.append(0)
            print(len(data[0]))
            for i in range(1,len(data)):
                for j in range(len(data[0])):
                    #print(data[i][j])
                    for x in selected_columns:
                        if(x==data[0][j]):
                            data[i][j]=float(data[i][j])
                            columns[j].append(data[i][j])
            print(columns)
            df=pd.DataFrame(data)
            column_mean=df.mean()
            print(column_mean)
            # for i in range(len(columns)):
            #     if(columns[i]):
            #         column_mean[i]=np.mean(columns[i])
                    # for data[0] in selected_columns:
                    #     print(x)


    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")

file_name = sys.argv[1]
imported_file(file_name)