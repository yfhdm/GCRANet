
import os
import cv2

from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

def test(test_pred_list,test_gt_list):



    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    for i_test in range(len(test_pred_list)):
        gt = cv2.imread(test_gt_list[i_test],cv2.IMREAD_GRAYSCALE)
        pred =cv2.imread(test_pred_list[i_test],cv2.IMREAD_GRAYSCALE)

        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        M.step(pred=pred, gt=gt)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    mae = M.get_results()["mae"]
    em=EM.get_results()["em"]

    return mae,wfm,sm,fm,em



if __name__ == "__main__":


    model_names= ["DACNet"]

    datasetnames=["CrackSeg9k"]
    for datasetname in datasetnames:

        for model_name in model_names:
            test_pred_dir=os.path.join("F:\\preds\\",model_name+"\\"+datasetname+"\\")

            test_gt_dir = os.path.join("D:\yanfeng\BASNet-master\SemanticData\process\\",datasetname+"\\test\\gt\\")
            test_gt_list = [os.path.join(test_gt_dir, file) for file in os.listdir(test_gt_dir)]

            test_pred_list = [os.path.join(test_pred_dir, file) for file in os.listdir(test_pred_dir)]
            test_gt_list = sorted(test_gt_list)

            test_pred_list = sorted(test_pred_list)


            mae,wfm,sm,fm,em= test(test_pred_list,test_gt_list)
            curr_results = {
                "model": model_name,
                "wFmeasure": '%.4f'% wfm,
                "Smeasure": '%.4f' % sm,
                "meanFm": '%.4f' % fm["curve"].mean(),
                "meanEm": '%.4f'%em["curve"].mean(),
                "MAE": '%.4f'% mae,
            }
            print(curr_results)

