import Performence as P
from read_initial_data import read_data

NGEN = 100

if __name__ == "__main__":
    # LR
    path_gp_1 = ".\\initial_dataset\\2\\"
    name_list = P.os.listdir(path_gp_1)
    for dataname in name_list:
        # 共用一个Test
        Train, Test = read_data(path_gp_1 + dataname, 1)
        file = dataname.rstrip('.dat')
        less = 'positive'
        more = 'negative'
        result_gmean = P.pd.DataFrame()
        # GP
        path_gp = ".\\GP_dataset\\" + file + "\\produce_set\\" + str(NGEN) + '\\'
        file_list = P.file_name(path_gp)
        G_Mean = []
        Auc = []
        randomseed = []
        parameter_c = []
        parameter_max_iter = []
        f1 = []
        num_gp = 0
        sum_gmean_gp = 0
        sum_AUC_gp = 0
        sum_f1 = 0
        for name in file_list:
            Train = P.pd.read_excel(path_gp + name, header=None)
            s, a, c, max_iter, F1 = P.LR_Performance(Train, Test, less, more)
            G_Mean.append(s)
            randomseed.append(num_gp)
            Auc.append(a)
            parameter_c.append(c)
            f1.append(F1)
            parameter_max_iter.append(max_iter)
            num_gp = num_gp + 1
            sum_gmean_gp += s
            sum_AUC_gp += a
            sum_f1 += F1
        G_mean_average = sum_gmean_gp / num_gp
        Auc_average = sum_AUC_gp / num_gp
        F1_average = sum_f1 / num_gp
        result_gmean['random seed'] = randomseed
        result_gmean['GP_Gmean'] = G_Mean
        result_gmean['GP_c'] = parameter_c
        result_gmean['GP_max_iter'] = parameter_max_iter
        result_gmean.loc[0, 'GP_Gmean_average'] = G_mean_average
        result_gmean['GP_AUC'] = Auc
        result_gmean.loc[0, 'GP_AUC_average'] = Auc_average
        result_gmean['GP_F1score'] = f1
        result_gmean.loc[0, 'GP_F1score_average'] = F1_average

        print("LR_GP: " + str(Auc_average))

        # Initial Data
        Train, Test = read_data(path_gp_1 + dataname, 1)
        feat_labels = [x for x in range(Train.shape[1])]
        Train.columns = feat_labels
        G_Mean, Auc, c, max_iter, F1 = P.LR_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'Initial_Gmean'] = G_Mean
        result_gmean.loc[0, 'Initial_AUC'] = Auc
        result_gmean.loc[0, 'Initial_F1'] = F1
        result_gmean.loc[0, 'Initial__c'] = c
        result_gmean.loc[0, 'Initial_max_iter'] = max_iter
        print("原始数据: " + str(Auc))
        # XLSX
        # SMOTE
        Train = P.pd.read_excel(".\\SMOTE_dataset\\SMOTE_" + file + ".xlsx", header=None)
        G_Mean, Auc, c, max_iter, F1 = P.LR_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'SMOTE_F1'] = F1
        result_gmean.loc[0, 'SMOTE_c'] = c
        result_gmean.loc[0, 'SMOTE_max_iter'] = max_iter
        print("SMOTE: " + str(Auc))

        # ADASYN
        Train = P.pd.read_excel(".\\ADASYN_dataset\\ADASYN_" + file + ".xlsx", header=None)
        G_Mean, Auc, c, max_iter, F1 = P.LR_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'ADASYN_Gmean'] = G_Mean
        result_gmean.loc[0, 'ADASYN_AUC'] = Auc
        result_gmean.loc[0, 'ADASYN_F1'] = F1
        result_gmean.loc[0, 'ADASYN_c'] = c
        result_gmean.loc[0, 'ADASYN_max_iter'] = max_iter
        print("ADASYN: " + str(Auc))

        # B1SMOTE
        Train = P.pd.read_excel(".\\Bordeline1_dataset\\SMOTEB1_" + file + ".xlsx", header=None)
        G_Mean, Auc, c, max_iter, F1 = P.LR_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B1SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B1SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B1SMOTE_F1'] = F1
        result_gmean.loc[0, 'B1SMOTE_c'] = c
        result_gmean.loc[0, 'B1SMOTE_max_iter'] = max_iter
        print("B1SMOTE: " + str(Auc))

        # B2SMOTE
        Train = P.pd.read_excel(".\\Bordeline2_dataset\\SMOTEB2_" + file + ".xlsx", header=None)
        G_Mean, Auc, c, max_iter, F1 = P.LR_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B2SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B2SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B2SMOTE_F1'] = F1
        result_gmean.loc[0, 'B2SMOTE_c'] = c
        result_gmean.loc[0, 'B2SMOTE_max_iter'] = max_iter
        print("B2SMOTE: " + str(Auc))

        # SMOTEENN
        Train = P.pd.read_excel(".\\SMOTE+ENN_dataset\\SMOTEENN_" + file + ".xlsx", header=None)
        G_Mean, Auc, c, max_iter, F1 = P.LR_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTEENN_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTEENN_AUC'] = Auc
        result_gmean.loc[0, 'SMOTEENN_F1'] = F1
        result_gmean.loc[0, 'SMOTEENN_c'] = c
        result_gmean.loc[0, 'SMOTEENN_max_iter'] = max_iter
        print("SMOTEENN: " + str(Auc))

        # SMOTETomek
        Train = P.pd.read_excel(".\\SMOTE+Tomek Links_dataset\\SMOTETomek_" + file + ".xlsx", header=None)
        G_Mean, Auc, c, max_iter, F1 = P.LR_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTETomek_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTETomek_AUC'] = Auc
        result_gmean.loc[0, 'SMOTETomek_F1'] = F1
        result_gmean.loc[0, 'SMOTETomek_c'] = c
        result_gmean.loc[0, 'SMOTETomek_max_iter'] = max_iter
        print("SMOTETomek: " + str(Auc))
        P.LR_save(result_gmean, file)

        # SVM
        result_gmean = P.pd.DataFrame()
        G_Mean = []
        Auc = []
        f1 = []
        randomseed = []
        parameter_c = []
        parameter_gamma = []
        parameter_kernel = []
        num_gp = 0
        sum_gmean_gp = 0
        sum_AUC_gp = 0
        sum_f1 = 0
        for name in file_list:
            Train = P.pd.read_excel(path_gp + name, header=None)
            s, a, kernel, C, gamma, F1 = P.SVM_Performance(Train, Test, less, more)
            G_Mean.append(s)
            randomseed.append(num_gp)
            Auc.append(a)
            f1.append(F1)
            num_gp = num_gp + 1
            sum_gmean_gp += s
            sum_AUC_gp += a
            parameter_c.append(C)
            parameter_gamma.append(gamma)
            parameter_kernel.append(kernel)
            sum_f1 += F1
        G_mean_average = sum_gmean_gp / num_gp
        Auc_average = sum_AUC_gp / num_gp
        F1_average = sum_f1 / num_gp

        result_gmean['random seed'] = randomseed
        result_gmean['GP_Gmean'] = G_Mean
        result_gmean['GP_kernel'] = parameter_kernel
        result_gmean['GP_c'] = parameter_c
        result_gmean['GP_gamma'] = parameter_gamma
        result_gmean.loc[0, 'GP_Gmean_average'] = G_mean_average
        result_gmean['GP_AUC'] = Auc
        result_gmean.loc[0, 'GP_AUC_average'] = Auc_average
        result_gmean['GP_F1score'] = f1
        result_gmean.loc[0, 'GP_F1score_average'] = F1_average
        print("SVM_GP: " + str(Auc_average))

        # Initial Data
        Train, Test = read_data(path_gp_1 + dataname, 1)
        feat_labels = [x for x in range(Train.shape[1])]
        Train.columns = feat_labels
        G_Mean, Auc, kernel, C, gamma, F1 = P.SVM_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'Initial_Gmean'] = G_Mean
        result_gmean.loc[0, 'Initial_AUC'] = Auc
        result_gmean.loc[0, 'Initial_F1'] = F1
        result_gmean.loc[0, 'Initial_kernel'] = kernel
        result_gmean.loc[0, 'Initial_parameter_c'] = C
        result_gmean.loc[0, 'Initial_parameter_gamma'] = gamma
        print("原始数据: " + str(Auc))

        # XLSX
        # SMOTE
        Train = P.pd.read_excel(".\\SMOTE_dataset\\SMOTE_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.SVM_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'SMOTE_F1'] = F1
        result_gmean.loc[0, 'SMOTE_kernel'] = kernel
        result_gmean.loc[0, 'SMOTE_parameter_c'] = C
        result_gmean.loc[0, 'SMOTE_parameter_gamma'] = gamma
        print("SMOTE: " + str(Auc))

        # ADASYN
        Train = P.pd.read_excel(".\\ADASYN_dataset\\ADASYN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.SVM_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'ADASYN_Gmean'] = G_Mean
        result_gmean.loc[0, 'ADASYN_AUC'] = Auc
        result_gmean.loc[0, 'ADASYN_F1'] = F1
        result_gmean.loc[0, 'ADASYN_kernel'] = kernel
        result_gmean.loc[0, 'ADASYN_parameter_c'] = C
        result_gmean.loc[0, 'ADASYN_parameter_gamma'] = gamma
        print("ADASYN: " + str(Auc))

        # B1SMOTE
        Train = P.pd.read_excel(".\\Bordeline1_dataset\\SMOTEB1_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.SVM_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B1SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B1SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B1SMOTE_F1'] = F1
        result_gmean.loc[0, 'B1SMOTE_kernel'] = kernel
        result_gmean.loc[0, 'B1SMOTE_parameter_c'] = C
        result_gmean.loc[0, 'B1SMOTE_parameter_gamma'] = gamma
        print("B1SMOTE: " + str(Auc))

        # B2SMOTE
        Train = P.pd.read_excel(".\\Bordeline2_dataset\\SMOTEB2_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.SVM_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B2SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B2SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B2SMOTE_F1'] = F1
        result_gmean.loc[0, 'B2SMOTE_kernel'] = kernel
        result_gmean.loc[0, 'B2SMOTE_parameter_c'] = C
        result_gmean.loc[0, 'B2SMOTE_parameter_gamma'] = gamma
        print("B2SMOTE: " + str(Auc))

        # SMOTEENN
        Train = P.pd.read_excel(".\\SMOTE+ENN_dataset\\SMOTEENN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.SVM_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTEENN_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTEENN_AUC'] = Auc
        result_gmean.loc[0, 'SMOTEENN_F1'] = F1
        result_gmean.loc[0, 'SMOTEENN_kernel'] = kernel
        result_gmean.loc[0, 'SMOTEENN_parameter_c'] = C
        result_gmean.loc[0, 'SMOTEENN_parameter_gamma'] = gamma
        print("SMOTEENN: " + str(Auc))

        # SMOTETomek
        Train = P.pd.read_excel(".\\SMOTE+Tomek Links_dataset\\SMOTETomek_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.SVM_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTETomek_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTETomek_AUC'] = Auc
        result_gmean.loc[0, 'SMOTETomek_F1'] = F1
        result_gmean.loc[0, 'SMOTETomek_kernel'] = kernel
        result_gmean.loc[0, 'SMOTETomek_parameter_c'] = C
        result_gmean.loc[0, 'SMOTETomek_parameter_gamma'] = gamma
        print("SMOTETomek: " + str(Auc))
        P.SVM_save(result_gmean, file)

        # DT
        result_gmean = P.pd.DataFrame()
        G_Mean = []
        Auc = []
        randomseed = []
        parameter_c = []
        parameter_gamma = []
        parameter_kernel = []
        f1 = []

        num_gp = 0
        sum_gmean_gp = 0
        sum_AUC_gp = 0
        sum_f1 = 0
        for name in file_list:
            Train = P.pd.read_excel(path_gp + name, header=None)
            s, a, kernel, C, gamma, F1 = P.DT_Performance(Train, Test, less, more)
            G_Mean.append(s)
            randomseed.append(num_gp)
            Auc.append(a)
            f1.append(F1)
            num_gp = num_gp + 1
            sum_gmean_gp += s
            sum_AUC_gp += a
            sum_f1 += F1
            parameter_c.append(C)
            parameter_gamma.append(gamma)
            parameter_kernel.append(kernel)

        G_mean_average = sum_gmean_gp / num_gp
        Auc_average = sum_AUC_gp / num_gp
        F1_average = sum_f1 / num_gp

        result_gmean['random seed'] = randomseed
        result_gmean['GP_Gmean'] = G_Mean
        result_gmean['GP_max_depth'] = parameter_kernel
        result_gmean['GP_min_samples_leaf'] = parameter_c
        result_gmean['GP_min_impurity_decrease'] = parameter_gamma
        result_gmean.loc[0, 'GP_Gmean_average'] = G_mean_average
        # result_gmean['GP_Gmean_average'][0] = G_mean_average
        result_gmean['GP_AUC'] = Auc
        result_gmean.loc[0, 'GP_AUC_average'] = Auc_average
        result_gmean['GP_F1score'] = f1
        result_gmean.loc[0, 'GP_F1score_average'] = F1_average
        print("DT_GP: " + str(Auc_average))

        # Initial Data
        Train, Test = read_data(path_gp_1 + dataname, 1)
        feat_labels = [x for x in range(Train.shape[1])]
        Train.columns = feat_labels
        G_Mean, Auc, kernel, C, gamma, F1 = P.DT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'Initial_Gmean'] = G_Mean
        result_gmean.loc[0, 'Initial_AUC'] = Auc
        result_gmean.loc[0, 'Initial_F1'] = F1
        result_gmean.loc[0, 'Initial_max_depth'] = kernel
        result_gmean.loc[0, 'Initial_min_samples_leaf'] = C
        result_gmean.loc[0, 'Initial_min_impurity_decrease'] = gamma
        print("原始数据: " + str(Auc))

        # XLSX
        # SMOTE
        Train = P.pd.read_excel(".\\SMOTE_dataset\\SMOTE_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.DT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'SMOTE_F1'] = F1
        result_gmean.loc[0, 'SMOTE_max_depth'] = kernel
        result_gmean.loc[0, 'SMOTE_min_samples_leaf'] = C
        result_gmean.loc[0, 'SMOTE_min_impurity_decrease'] = gamma
        print("SMOTE: " + str(Auc))

        # ADASYN
        Train = P.pd.read_excel(".\\ADASYN_dataset\\ADASYN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.DT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'ADASYN_Gmean'] = G_Mean
        result_gmean.loc[0, 'ADASYN_AUC'] = Auc
        result_gmean.loc[0, 'ADASYN_F1'] = F1
        result_gmean.loc[0, 'ADASYN_max_depth'] = kernel
        result_gmean.loc[0, 'ADASYN_min_samples_leaf'] = C
        result_gmean.loc[0, 'ADASYN_min_impurity_decrease'] = gamma
        print("ADASYN: " + str(Auc))

        # B1SMOTE
        Train = P.pd.read_excel(".\\Bordeline1_dataset\\SMOTEB1_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.DT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B1SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B1SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B1SMOTE_F1'] = F1
        result_gmean.loc[0, 'B1SMOTE_max_depth'] = kernel
        result_gmean.loc[0, 'B1SMOTE_min_samples_leaf'] = C
        result_gmean.loc[0, 'B1SMOTE_min_impurity_decrease'] = gamma
        print("B1SMOTE: " + str(Auc))

        # B2SMOTE
        Train = P.pd.read_excel(".\\Bordeline2_dataset\\SMOTEB2_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.DT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B2SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B2SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B2SMOTE_F1'] = F1
        result_gmean.loc[0, 'B2SMOTE_max_depth'] = kernel
        result_gmean.loc[0, 'B2SMOTE_min_samples_leaf'] = C
        result_gmean.loc[0, 'B2SMOTE_min_impurity_decrease'] = gamma
        print("B2SMOTE: " + str(Auc))

        # SMOTEENN
        Train = P.pd.read_excel(".\\SMOTE+ENN_dataset\\SMOTEENN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.DT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTEENN_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTEENN_AUC'] = Auc
        result_gmean.loc[0, 'SMOTEENN_F1'] = F1
        result_gmean.loc[0, 'SMOTEENN_max_depth'] = kernel
        result_gmean.loc[0, 'SMOTEENN_min_samples_leaf'] = C
        result_gmean.loc[0, 'SMOTEENN_min_impurity_decrease'] = gamma
        print("SMOTEENN: " + str(Auc))

        # SMOTETomek
        Train = P.pd.read_excel(".\\SMOTE+Tomek Links_dataset\\SMOTETomek_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.DT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTETomek_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTETomek_AUC'] = Auc
        result_gmean.loc[0, 'SMOTETomek_F1'] = F1
        result_gmean.loc[0, 'SMOTETomek_max_depth'] = kernel
        result_gmean.loc[0, 'SMOTETomek_min_samples_leaf'] = C
        result_gmean.loc[0, 'SMOTETomek_min_impurity_decrease'] = gamma
        print("SMOTETomek: " + str(Auc))
        P.DT_save(result_gmean, file)

        # RF
        result_gmean = P.pd.DataFrame()
        G_Mean = []
        Auc = []
        randomseed = []
        parameter_c = []
        parameter_gamma = []
        parameter_kernel = []
        f1 = []
        num_gp = 0
        sum_gmean_gp = 0
        sum_AUC_gp = 0
        sum_f1 = 0

        for name in file_list:
            Train = P.pd.read_excel(path_gp + name, header=None)
            s, a, kernel, C, gamma, F1 = P.RF_Performance(Train, Test, less, more)
            G_Mean.append(s)
            randomseed.append(num_gp)
            Auc.append(a)
            f1.append(F1)
            num_gp = num_gp + 1
            sum_gmean_gp += s
            sum_AUC_gp += a
            sum_f1 += F1
            parameter_c.append(C)
            parameter_gamma.append(gamma)
            parameter_kernel.append(kernel)

        G_mean_average = sum_gmean_gp / num_gp
        Auc_average = sum_AUC_gp / num_gp
        F1_average = sum_f1 / num_gp

        result_gmean['random seed'] = randomseed
        result_gmean['GP_Gmean'] = G_Mean
        result_gmean['GP_n_estimators'] = parameter_kernel
        result_gmean['GP_max_depth'] = parameter_c
        result_gmean['GP_min_samples_split'] = parameter_gamma
        result_gmean.loc[0, 'GP_Gmean_average'] = G_mean_average
        # result_gmean['GP_Gmean_average'][0] = G_mean_average
        result_gmean['GP_AUC'] = Auc
        result_gmean.loc[0, 'GP_AUC_average'] = Auc_average
        result_gmean['GP_F1score'] = f1
        result_gmean.loc[0, 'GP_F1score_average'] = F1_average
        print("RF_GP: " + str(Auc_average))

        # Initial Data
        Train, Test = read_data(path_gp_1 + dataname, 1)
        feat_labels = [x for x in range(Train.shape[1])]
        Train.columns = feat_labels
        G_Mean, Auc, kernel, C, gamma, F1 = P.RF_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'Initial_Gmean'] = G_Mean
        result_gmean.loc[0, 'Initial_AUC'] = Auc
        result_gmean.loc[0, 'Initial_F1'] = F1
        result_gmean.loc[0, 'Initial_n_estimators'] = kernel
        result_gmean.loc[0, 'Initial_max_depth'] = C
        result_gmean.loc[0, 'Initial_min_samples_split'] = gamma
        print("原始数据: " + str(Auc))

        # XLSX
        # SMOTE
        Train = P.pd.read_excel(".\\SMOTE_dataset\\SMOTE_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.RF_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'SMOTE_F1'] = F1
        result_gmean.loc[0, 'SMOTE_n_estimators'] = kernel
        result_gmean.loc[0, 'SMOTE_max_depth'] = C
        result_gmean.loc[0, 'SMOTE_min_samples_split'] = gamma
        print("SMOTE: " + str(Auc))

        # ADASYN
        Train = P.pd.read_excel(".\\ADASYN_dataset\\ADASYN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.RF_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'ADASYN_Gmean'] = G_Mean
        result_gmean.loc[0, 'ADASYN_AUC'] = Auc
        result_gmean.loc[0, 'ADASYN_F1'] = F1
        result_gmean.loc[0, 'ADASYN_n_estimators'] = kernel
        result_gmean.loc[0, 'ADASYN_max_depth'] = C
        result_gmean.loc[0, 'ADASYN_min_samples_split'] = gamma
        print("ADASYN: " + str(Auc))

        # B1SMOTE
        Train = P.pd.read_excel(".\\Bordeline1_dataset\\SMOTEB1_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.RF_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B1SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B1SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B1SMOTE_F1'] = F1
        result_gmean.loc[0, 'B1SMOTE_n_estimators'] = kernel
        result_gmean.loc[0, 'B1SMOTE_max_depth'] = C
        result_gmean.loc[0, 'B1SMOTE_min_impurity_decrease'] = gamma
        print("B1SMOTE: " + str(Auc))

        # B2SMOTE
        Train = P.pd.read_excel(".\\Bordeline2_dataset\\SMOTEB2_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.RF_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B2SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B2SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B2SMOTE_F1'] = F1
        result_gmean.loc[0, 'B2SMOTE_n_estimators'] = kernel
        result_gmean.loc[0, 'B2SMOTE_max_depth'] = C
        result_gmean.loc[0, 'B2SMOTE_min_samples_split'] = gamma
        print("B2SMOTE: " + str(Auc))

        # SMOTEENN
        Train = P.pd.read_excel(".\\SMOTE+ENN_dataset\\SMOTEENN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.RF_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTEENN_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTEENN_AUC'] = Auc
        result_gmean.loc[0, 'SMOTEENN_F1'] = F1
        result_gmean.loc[0, 'SMOTEENN_n_estimators'] = kernel
        result_gmean.loc[0, 'SMOTEENN_max_depth'] = C
        result_gmean.loc[0, 'SMOTEENN_min_samples_split'] = gamma
        print("SMOTEENN: " + str(Auc))

        # SMOTETomek
        Train = P.pd.read_excel(".\\SMOTE+Tomek Links_dataset\\SMOTETomek_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.RF_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTETomek_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTETomek_AUC'] = Auc
        result_gmean.loc[0, 'SMOTETomek_F1'] = F1
        result_gmean.loc[0, 'SMOTETomek_n_estimators'] = kernel
        result_gmean.loc[0, 'SMOTETomek_max_depth'] = C
        result_gmean.loc[0, 'SMOTETomek_min_samples_split'] = gamma
        print("SMOTETomek: " + str(Auc))
        P.RF_save(result_gmean, file)

        # GBDT
        result_gmean = P.pd.DataFrame()
        G_Mean = []
        Auc = []
        randomseed = []
        parameter_c = []
        parameter_gamma = []
        parameter_kernel = []
        f1 = []
        num_gp = 0
        sum_gmean_gp = 0
        sum_AUC_gp = 0
        sum_f1 = 0

        for name in file_list:
            Train = P.pd.read_excel(path_gp + name, header=None)
            s, a, kernel, C, gamma, F1 = P.GBDT_Performance(Train, Test, less, more)
            G_Mean.append(s)
            randomseed.append(num_gp)
            Auc.append(a)
            f1.append(F1)
            num_gp = num_gp + 1
            sum_gmean_gp += s
            sum_AUC_gp += a
            sum_f1 += F1
            parameter_c.append(C)
            parameter_gamma.append(gamma)
            parameter_kernel.append(kernel)

        G_mean_average = sum_gmean_gp / num_gp
        Auc_average = sum_AUC_gp / num_gp
        F1_average = sum_f1 / num_gp

        result_gmean['random seed'] = randomseed
        result_gmean['GP_Gmean'] = G_Mean
        result_gmean['GP_n_estimators'] = parameter_kernel
        result_gmean['GP_max_depth'] = parameter_c
        result_gmean['GP_min_samples_split'] = parameter_gamma
        result_gmean.loc[0, 'GP_Gmean_average'] = G_mean_average
        # result_gmean['GP_Gmean_average'][0] = G_mean_average
        result_gmean['GP_AUC'] = Auc
        result_gmean.loc[0, 'GP_AUC_average'] = Auc_average
        result_gmean['GP_F1score'] = f1
        result_gmean.loc[0, 'GP_F1score_average'] = F1_average
        print("GBDT_GP: " + str(Auc_average))

        # Initial Data
        Train, Test = read_data(path_gp_1 + dataname, 1)
        feat_labels = [x for x in range(Train.shape[1])]
        Train.columns = feat_labels
        G_Mean, Auc, kernel, C, gamma, F1 = P.GBDT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'Initial_Gmean'] = G_Mean
        result_gmean.loc[0, 'Initial_AUC'] = Auc
        result_gmean.loc[0, 'Initial_F1'] = F1
        result_gmean.loc[0, 'Initial_n_estimators'] = kernel
        result_gmean.loc[0, 'Initial_max_depth'] = C
        result_gmean.loc[0, 'Initial_min_samples_split'] = gamma
        print("原始数据: " + str(Auc))

        # XLSX
        # SMOTE
        Train = P.pd.read_excel(".\\SMOTE_dataset\\SMOTE_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.GBDT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'SMOTE_F1'] = F1
        result_gmean.loc[0, 'SMOTE_n_estimators'] = kernel
        result_gmean.loc[0, 'SMOTE_max_depth'] = C
        result_gmean.loc[0, 'SMOTE_min_samples_split'] = gamma
        print("SMOTE: " + str(Auc))

        # ADASYN
        Train = P.pd.read_excel(".\\ADASYN_dataset\\ADASYN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.GBDT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'ADASYN_Gmean'] = G_Mean
        result_gmean.loc[0, 'ADASYN_AUC'] = Auc
        result_gmean.loc[0, 'ADASYN_F1'] = F1
        result_gmean.loc[0, 'ADASYN_n_estimators'] = kernel
        result_gmean.loc[0, 'ADASYN_max_depth'] = C
        result_gmean.loc[0, 'ADASYN_min_samples_split'] = gamma
        print("ADASYN: " + str(Auc))

        # B1SMOTE
        Train = P.pd.read_excel(".\\Bordeline1_dataset\\SMOTEB1_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.GBDT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B1SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B1SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B1SMOTE_F1'] = F1
        result_gmean.loc[0, 'B1SMOTE_n_estimators'] = kernel
        result_gmean.loc[0, 'B1SMOTE_max_depth'] = C
        result_gmean.loc[0, 'B1SMOTE_min_impurity_decrease'] = gamma
        print("B1SMOTE: " + str(Auc))

        # B2SMOTE
        Train = P.pd.read_excel(".\\Bordeline2_dataset\\SMOTEB2_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.GBDT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B2SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B2SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B2SMOTE_F1'] = F1
        result_gmean.loc[0, 'B2SMOTE_n_estimators'] = kernel
        result_gmean.loc[0, 'B2SMOTE_max_depth'] = C
        result_gmean.loc[0, 'B2SMOTE_min_samples_split'] = gamma
        print("B2SMOTE: " + str(Auc))

        # SMOTEENN
        Train = P.pd.read_excel(".\\SMOTE+ENN_dataset\\SMOTEENN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.GBDT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTEENN_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTEENN_AUC'] = Auc
        result_gmean.loc[0, 'SMOTEENN_F1'] = F1
        result_gmean.loc[0, 'SMOTEENN_n_estimators'] = kernel
        result_gmean.loc[0, 'SMOTEENN_max_depth'] = C
        result_gmean.loc[0, 'SMOTEENN_min_samples_split'] = gamma
        print("SMOTEENN: " + str(Auc))

        # SMOTETomek
        Train = P.pd.read_excel(".\\SMOTE+Tomek Links_dataset\\SMOTETomek_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, C, gamma, F1 = P.GBDT_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTETomek_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTETomek_AUC'] = Auc
        result_gmean.loc[0, 'SMOTETomek_F1'] = F1
        result_gmean.loc[0, 'SMOTETomek_n_estimators'] = kernel
        result_gmean.loc[0, 'SMOTETomek_max_depth'] = C
        result_gmean.loc[0, 'SMOTETomek_min_samples_split'] = gamma
        print("SMOTETomek: " + str(Auc))
        P.GBDT_save(result_gmean, file)

        # KNN
        result_gmean = P.pd.DataFrame()
        G_Mean = []
        Auc = []
        randomseed = []
        parameter_c = []
        parameter_gamma = []
        parameter_kernel = []
        f1 = []
        num_gp = 0
        sum_gmean_gp = 0
        sum_AUC_gp = 0
        sum_f1 = 0

        for name in file_list:
            Train = P.pd.read_excel(path_gp + name, header=None)
            s, a, kernel, F1 = P.Knn_Performance(Train, Test, less, more)
            G_Mean.append(s)
            randomseed.append(num_gp)
            Auc.append(a)
            f1.append(F1)
            num_gp = num_gp + 1
            sum_gmean_gp += s
            sum_AUC_gp += a
            sum_f1 += F1
            parameter_kernel.append(kernel)

        G_mean_average = sum_gmean_gp / num_gp
        Auc_average = sum_AUC_gp / num_gp
        F1_average = sum_f1 / num_gp

        result_gmean['random seed'] = randomseed
        result_gmean['GP_Gmean'] = G_Mean
        result_gmean['GP_k'] = parameter_kernel
        result_gmean.loc[0, 'GP_Gmean_average'] = G_mean_average
        # result_gmean['GP_Gmean_average'][0] = G_mean_average
        result_gmean['GP_AUC'] = Auc
        result_gmean.loc[0, 'GP_AUC_average'] = Auc_average
        result_gmean['GP_F1score'] = f1
        result_gmean.loc[0, 'GP_F1score_average'] = F1_average
        print("KNN_GP: " + str(Auc_average))

        # Initial Data
        Train, Test = read_data(path_gp_1 + dataname, 1)
        feat_labels = [x for x in range(Train.shape[1])]
        Train.columns = feat_labels
        G_Mean, Auc, kernel, F1 = P.Knn_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'Initial_Gmean'] = G_Mean
        result_gmean.loc[0, 'Initial_AUC'] = Auc
        result_gmean.loc[0, 'Initial_F1'] = F1
        result_gmean.loc[0, 'Initial_k'] = kernel
        print("原始数据: " + str(Auc))

        # XLSX
        # SMOTE
        Train = P.pd.read_excel(".\\SMOTE_dataset\\SMOTE_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.Knn_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'SMOTE_F1'] = F1
        result_gmean.loc[0, 'SMOTE_k'] = kernel
        print("SMOTE: " + str(Auc))

        # ADASYN
        Train = P.pd.read_excel(".\\ADASYN_dataset\\ADASYN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.Knn_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'ADASYN_Gmean'] = G_Mean
        result_gmean.loc[0, 'ADASYN_AUC'] = Auc
        result_gmean.loc[0, 'ADASYN_F1'] = F1
        result_gmean.loc[0, 'ADASYN_k'] = kernel
        print("ADASYN: " + str(Auc))

        # B1SMOTE
        Train = P.pd.read_excel(".\\Bordeline1_dataset\\SMOTEB1_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.Knn_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B1SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B1SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B1SMOTE_F1'] = F1
        result_gmean.loc[0, 'B1SMOTE_k'] = kernel
        print("B1SMOTE: " + str(Auc))

        # B2SMOTE
        Train = P.pd.read_excel(".\\Bordeline2_dataset\\SMOTEB2_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.Knn_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B2SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B2SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B2SMOTE_F1'] = F1
        result_gmean.loc[0, 'B2SMOTE_k'] = kernel
        print("B2SMOTE: " + str(Auc))

        # SMOTEENN
        Train = P.pd.read_excel(".\\SMOTE+ENN_dataset\\SMOTEENN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.Knn_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTEENN_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTEENN_AUC'] = Auc
        result_gmean.loc[0, 'SMOTEENN_F1'] = F1
        result_gmean.loc[0, 'SMOTEENN_k'] = kernel
        print("SMOTEENN: " + str(Auc))

        # SMOTETomek
        Train = P.pd.read_excel(".\\SMOTE+Tomek Links_dataset\\SMOTETomek_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.Knn_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTETomek_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTETomek_AUC'] = Auc
        result_gmean.loc[0, 'SMOTETomek_F1'] = F1
        result_gmean.loc[0, 'SMOTETomek_k'] = kernel
        print("SMOTETomek: " + str(Auc))
        P.KNN_save(result_gmean, file)

        # GNB
        result_gmean = P.pd.DataFrame()
        G_Mean = []
        Auc = []
        randomseed = []
        parameter_c = []
        parameter_gamma = []
        parameter_kernel = []
        f1 = []
        num_gp = 0
        sum_gmean_gp = 0
        sum_AUC_gp = 0
        sum_f1 = 0

        for name in file_list:
            Train = P.pd.read_excel(path_gp + name, header=None)
            s, a, F1 = P.GNB_Performance(Train, Test, less, more)
            G_Mean.append(s)
            randomseed.append(num_gp)
            Auc.append(a)
            f1.append(F1)
            num_gp = num_gp + 1
            sum_gmean_gp += s
            sum_AUC_gp += a
            sum_f1 += F1


        G_mean_average = sum_gmean_gp / num_gp
        Auc_average = sum_AUC_gp / num_gp
        F1_average = sum_f1 / num_gp

        result_gmean['random seed'] = randomseed
        result_gmean['GP_Gmean'] = G_Mean
        result_gmean.loc[0, 'GP_Gmean_average'] = G_mean_average
        # result_gmean['GP_Gmean_average'][0] = G_mean_average
        result_gmean['GP_AUC'] = Auc
        result_gmean.loc[0, 'GP_AUC_average'] = Auc_average
        result_gmean['GP_F1score'] = f1
        result_gmean.loc[0, 'GP_F1score_average'] = F1_average
        print("GNB_GP: " + str(Auc_average))

        # Initial Data
        Train, Test = read_data(path_gp_1 + dataname, 1)
        feat_labels = [x for x in range(Train.shape[1])]
        Train.columns = feat_labels
        G_Mean, Auc, F1 = P.GNB_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'Initial_Gmean'] = G_Mean
        result_gmean.loc[0, 'Initial_AUC'] = Auc
        result_gmean.loc[0, 'Initial_F1'] = F1
        print("原始数据: " + str(Auc))

        # XLSX
        # SMOTE
        Train = P.pd.read_excel(".\\SMOTE_dataset\\SMOTE_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.GNB_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'SMOTE_F1'] = F1
        print("SMOTE: " + str(Auc))

        # ADASYN
        Train = P.pd.read_excel(".\\ADASYN_dataset\\ADASYN_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.GNB_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'ADASYN_Gmean'] = G_Mean
        result_gmean.loc[0, 'ADASYN_AUC'] = Auc
        result_gmean.loc[0, 'ADASYN_F1'] = F1
        print("ADASYN: " + str(Auc))

        # B1SMOTE
        Train = P.pd.read_excel(".\\Bordeline1_dataset\\SMOTEB1_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.GNB_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B1SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B1SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B1SMOTE_F1'] = F1
        print("B1SMOTE: " + str(Auc))

        # B2SMOTE
        Train = P.pd.read_excel(".\\Bordeline2_dataset\\SMOTEB2_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.GNB_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B2SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B2SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B2SMOTE_F1'] = F1
        print("B2SMOTE: " + str(Auc))

        # SMOTEENN
        Train = P.pd.read_excel(".\\SMOTE+ENN_dataset\\SMOTEENN_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.GNB_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTEENN_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTEENN_AUC'] = Auc
        result_gmean.loc[0, 'SMOTEENN_F1'] = F1
        print("SMOTEENN: " + str(Auc))

        # SMOTETomek
        Train = P.pd.read_excel(".\\SMOTE+Tomek Links_dataset\\SMOTETomek_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.GNB_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTETomek_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTETomek_AUC'] = Auc
        result_gmean.loc[0, 'SMOTETomek_F1'] = F1
        print("SMOTETomek: " + str(Auc))
        P.GNB_save(result_gmean, file)

        # MLP
        result_gmean = P.pd.DataFrame()
        G_Mean = []
        Auc = []
        randomseed = []
        parameter_c = []
        parameter_gamma = []
        parameter_kernel = []
        f1 = []
        num_gp = 0
        sum_gmean_gp = 0
        sum_AUC_gp = 0
        sum_f1 = 0

        for name in file_list:
            Train = P.pd.read_excel(path_gp + name, header=None)
            s, a, F1 = P.MLP_Performance(Train, Test, less, more)
            G_Mean.append(s)
            randomseed.append(num_gp)
            Auc.append(a)
            f1.append(F1)
            num_gp = num_gp + 1
            sum_gmean_gp += s
            sum_AUC_gp += a
            sum_f1 += F1

        G_mean_average = sum_gmean_gp / num_gp
        Auc_average = sum_AUC_gp / num_gp
        F1_average = sum_f1 / num_gp

        result_gmean['random seed'] = randomseed
        result_gmean['GP_Gmean'] = G_Mean
        result_gmean.loc[0, 'GP_Gmean_average'] = G_mean_average
        # result_gmean['GP_Gmean_average'][0] = G_mean_average
        result_gmean['GP_AUC'] = Auc
        result_gmean.loc[0, 'GP_AUC_average'] = Auc_average
        result_gmean['GP_F1score'] = f1
        result_gmean.loc[0, 'GP_F1score_average'] = F1_average
        print("MLP_GP: " + str(Auc_average))

        # Initial Data
        Train, Test = read_data(path_gp_1 + dataname, 1)
        feat_labels = [x for x in range(Train.shape[1])]
        Train.columns = feat_labels
        G_Mean, Auc, F1 = P.MLP_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'Initial_Gmean'] = G_Mean
        result_gmean.loc[0, 'Initial_AUC'] = Auc
        result_gmean.loc[0, 'Initial_F1'] = F1
        print("原始数据: " + str(Auc))

        # XLSX
        # SMOTE
        Train = P.pd.read_excel(".\\SMOTE_dataset\\SMOTE_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.MLP_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'SMOTE_F1'] = F1
        print("SMOTE: " + str(Auc))

        # ADASYN
        Train = P.pd.read_excel(".\\ADASYN_dataset\\ADASYN_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.MLP_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'ADASYN_Gmean'] = G_Mean
        result_gmean.loc[0, 'ADASYN_AUC'] = Auc
        result_gmean.loc[0, 'ADASYN_F1'] = F1
        print("ADASYN: " + str(Auc))

        # B1SMOTE
        Train = P.pd.read_excel(".\\Bordeline1_dataset\\SMOTEB1_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.MLP_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B1SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B1SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B1SMOTE_F1'] = F1
        print("B1SMOTE: " + str(Auc))

        # B2SMOTE
        Train = P.pd.read_excel(".\\Bordeline2_dataset\\SMOTEB2_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.MLP_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B2SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B2SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B2SMOTE_F1'] = F1
        print("B2SMOTE: " + str(Auc))

        # SMOTEENN
        Train = P.pd.read_excel(".\\SMOTE+ENN_dataset\\SMOTEENN_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.MLP_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTEENN_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTEENN_AUC'] = Auc
        result_gmean.loc[0, 'SMOTEENN_F1'] = F1
        print("SMOTEENN: " + str(Auc))

        # SMOTETomek
        Train = P.pd.read_excel(".\\SMOTE+Tomek Links_dataset\\SMOTETomek_" + file + ".xlsx", header=None)
        G_Mean, Auc, F1 = P.MLP_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTETomek_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTETomek_AUC'] = Auc
        result_gmean.loc[0, 'SMOTETomek_F1'] = F1
        print("SMOTETomek: " + str(Auc))
        P.MLP_save(result_gmean, file)

        # ADA
        result_gmean = P.pd.DataFrame()
        G_Mean = []
        Auc = []
        randomseed = []
        parameter_c = []
        parameter_gamma = []
        parameter_kernel = []
        f1 = []
        num_gp = 0
        sum_gmean_gp = 0
        sum_AUC_gp = 0
        sum_f1 = 0

        for name in file_list:
            Train = P.pd.read_excel(path_gp + name, header=None)
            s, a, kernel, F1 = P.ADA_Performance(Train, Test, less, more)
            G_Mean.append(s)
            randomseed.append(num_gp)
            Auc.append(a)
            f1.append(F1)
            num_gp = num_gp + 1
            sum_gmean_gp += s
            sum_AUC_gp += a
            sum_f1 += F1
            parameter_kernel.append(kernel)

        G_mean_average = sum_gmean_gp / num_gp
        Auc_average = sum_AUC_gp / num_gp
        F1_average = sum_f1 / num_gp

        result_gmean['random seed'] = randomseed
        result_gmean['GP_Gmean'] = G_Mean
        result_gmean['GP_n_estimators'] = parameter_kernel
        result_gmean.loc[0, 'GP_Gmean_average'] = G_mean_average
        # result_gmean['GP_Gmean_average'][0] = G_mean_average
        result_gmean['GP_AUC'] = Auc
        result_gmean.loc[0, 'GP_AUC_average'] = Auc_average
        result_gmean['GP_F1score'] = f1
        result_gmean.loc[0, 'GP_F1score_average'] = F1_average
        print("ADA_GP: " + str(Auc_average))

        # Initial Data
        Train, Test = read_data(path_gp_1 + dataname, 1)
        feat_labels = [x for x in range(Train.shape[1])]
        Train.columns = feat_labels
        G_Mean, Auc, kernel, F1 = P.ADA_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'Initial_Gmean'] = G_Mean
        result_gmean.loc[0, 'Initial_AUC'] = Auc
        result_gmean.loc[0, 'Initial_F1'] = F1
        result_gmean.loc[0, 'Initial_n_estimators'] = kernel
        print("原始数据: " + str(Auc))

        # XLSX
        # SMOTE
        Train = P.pd.read_excel(".\\SMOTE_dataset\\SMOTE_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.ADA_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'SMOTE_F1'] = F1
        result_gmean.loc[0, 'SMOTE_n_estimators'] = kernel
        print("SMOTE: " + str(Auc))

        # ADASYN
        Train = P.pd.read_excel(".\\ADASYN_dataset\\ADASYN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.ADA_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'ADASYN_Gmean'] = G_Mean
        result_gmean.loc[0, 'ADASYN_AUC'] = Auc
        result_gmean.loc[0, 'ADASYN_F1'] = F1
        result_gmean.loc[0, 'ADASYN_n_estimators'] = kernel
        print("ADASYN: " + str(Auc))

        # B1SMOTE
        Train = P.pd.read_excel(".\\Bordeline1_dataset\\SMOTEB1_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.ADA_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B1SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B1SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B1SMOTE_F1'] = F1
        result_gmean.loc[0, 'B1SMOTE_n_estimators'] = kernel
        print("B1SMOTE: " + str(Auc))

        # B2SMOTE
        Train = P.pd.read_excel(".\\Bordeline2_dataset\\SMOTEB2_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.ADA_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'B2SMOTE_Gmean'] = G_Mean
        result_gmean.loc[0, 'B2SMOTE_AUC'] = Auc
        result_gmean.loc[0, 'B2SMOTE_F1'] = F1
        result_gmean.loc[0, 'B2SMOTE_n_estimators'] = kernel
        print("B2SMOTE: " + str(Auc))

        # SMOTEENN
        Train = P.pd.read_excel(".\\SMOTE+ENN_dataset\\SMOTEENN_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.ADA_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTEENN_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTEENN_AUC'] = Auc
        result_gmean.loc[0, 'SMOTEENN_F1'] = F1
        result_gmean.loc[0, 'SMOTEENN_n_estimators'] = kernel
        print("SMOTEENN: " + str(Auc))

        # SMOTETomek
        Train = P.pd.read_excel(".\\SMOTE+Tomek Links_dataset\\SMOTETomek_" + file + ".xlsx", header=None)
        G_Mean, Auc, kernel, F1 = P.ADA_Performance(Train, Test, less, more)
        result_gmean.loc[0, 'SMOTETomek_Gmean'] = G_Mean
        result_gmean.loc[0, 'SMOTETomek_AUC'] = Auc
        result_gmean.loc[0, 'SMOTETomek_F1'] = F1
        result_gmean.loc[0, 'SMOTETomek_n_estimators'] = kernel
        print("SMOTETomek: " + str(Auc))
        P.ADA_save(result_gmean, file)
