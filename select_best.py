import os
import pandas as pd


path = "/content/drive/MyDrive/data/swav/dumped_path/method2/"

if __name__ == "__main__":

    for root, dirs, files in os.walk(path, topdown=True):

        for f in files:

            # if dirs:
            #     continue

            if f == "log.csv":

                csv_file = pd.read_csv(root + "/" + f)

                val_acc = csv_file['val_acc'].values
                val_recall1 = csv_file['val_recall1'].values
                val_recallk = csv_file['val_recallk'].values

                best_acc = 0
                best_recall1 = 0
                best_recallk = 0

                for v_acc, v_recall1, v_recallk in zip(val_acc, val_recall1, val_recallk):

                    if v_acc > best_acc:
                        best_acc = v_acc

                        best_recall1 = v_recall1
                        best_recallk = v_recallk


                # print(root.split("/")[-1] + " - " + "best_acc: " + str(best_acc) + ", " + "recall1: " + str(best_recall1) + ", " + "recallk: " + str(best_recallk))
                print(root.split("/")[-1] + " - "  + str(best_acc) + ", " + str(best_recall1) + ", " + str(best_recallk))



