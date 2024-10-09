# binary N staging performance
DC_result <- read.xlsx('./SHANXI_patient_preds_LN_num.xlsx',1)
# IVC
IVC_result <- DC_result[which(DC_result$dataset=='val'),]
# binary N staging
IVC_label <- IVC_result$meta_status
IVC_preds <- IVC_result$patch_pred_max
IVC_auc <- roc(as.factor(IVC_label),IVC_preds,
              plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
IVC_auc$auc
metrics_calc_preds_label(as.factor(IVC_label),IVC_preds)
# ternary N staging
IVC_cN_pred <- IVC_result$cNstage
confusionMatrix(data=as.factor(IVC_cN_pred), reference=as.factor(IVC_result$N_stage), mode='everything')
rcorr.cens(as.numeric(IVC_cN_pred), as.factor(IVC_result$N_stage))[[1]]

# EVC1
EVC1_result <- read.xlsx('./FUDAN_patient_preds_LN_num.xlsx',1)
EVC1_label <- EVC1_result$meta_status
EVC1_preds <- EVC1_result$patch_pred_max
EVC1_auc <- roc(as.factor(EVC1_label),EVC1_preds,
                plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
EVC1_auc$auc
metrics_calc_preds_label(as.factor(EVC1_label),EVC1_preds)
# ternary N staging
EVC1_cN_pred <- EVC1_result$cNstage
confusionMatrix(data=as.factor(EVC1_cN_pred), reference=as.factor(EVC1_result$N_stage), mode='everything')
rcorr.cens(as.numeric(EVC1_cN_pred), as.factor(EVC1_result$N_stage))[[1]]


# EVC2
EVC2_result <- read.xlsx('./YUNNAN_patient_preds_LN_num.xlsx',1)
EVC2_label <- EVC2_result$meta_status
EVC2_preds <- EVC2_result$patch_pred_max
EVC2_auc <- roc(as.factor(EVC2_label),EVC2_preds,
                plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
EVC2_auc$auc
metrics_calc_preds_label(as.factor(EVC2_label),EVC2_preds)
# ternary N staging
EVC2_cN_pred <- EVC2_result$cNstage
confusionMatrix(data=as.factor(EVC2_cN_pred), reference=as.factor(EVC2_result$N_stage), mode='everything')
rcorr.cens(as.numeric(EVC2_cN_pred), as.factor(EVC2_result$N_stage))[[1]]


# OVC
OVC_label <- c(IVC_label, EVC1_label, EVC2_label)
OVC_preds <- c(IVC_preds, EVC1_preds, EVC2_preds)
OVC_auc <- roc(as.factor(OVC_label),OVC_preds,
              plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
OVC_auc$auc
metrics_calc_preds_label(as.factor(OVC_label),OVC_preds)
# ternary N staging
OVC_cN_pred <- c(IVC_cN_pred, EVC1_cN_pred, EVC2_cN_pred)
OVC_cN <- c(IVC_result$N_stage,EVC1_result$N_stage,EVC2_result$N_stage)
confusionMatrix(data=as.factor(OVC_cN_pred), reference=as.factor(OVC_cN), mode='everything')
rcorr.cens(as.numeric(OVC_cN_pred), as.factor(OVC_cN))[[1]]

