library(ggplot2)
library(glmnet)
library(caret)
library(pROC)
library(rms)
library(Hmisc)
library(xlsx)
library(e1071)
# calculate the accuracy sensitivity and specificity
func_acc_sen_spe <- function(label,predictions,cutoff,pos_label){
  right_num <- 0
  pred <- 0
  TP <- 0
  FP <- 0
  TN <- 0
  FN <- 0
  if(pos_label==1){
    for(ind in seq(length=length(label),from = 1,to=length(label))){
      if(predictions[ind]>cutoff){pred <- 1}else{pred <- 0}
      if(label[ind]==1 & pred==1){TP <- TP +1 }
      if(label[ind]==0 & pred==1){FP <- FP +1 }
      if(label[ind]==0 & pred==0){TN <- TN +1 }
      if(label[ind]==1 & pred==0){FN <- FN +1 }
    }
  }
  
  if(pos_label==0){
    for(ind in seq(length=length(label),from = 1,to=length(label))){
      if(predictions[ind]<cutoff){pred <- 1}else{pred <- 0}
      if(label[ind]==1 & pred==1){TP <- TP +1 }
      if(label[ind]==0 & pred==1){FP <- FP +1 }
      if(label[ind]==0 & pred==0){TN <- TN +1 }
      if(label[ind]==1 & pred==0){FN <- FN +1 }
    }
  }
  right_num <- TP + TN
  accuracy <- right_num/length(label)
  sensitivity <- TP/(TP+FN)
  specificity <- TN/(TN+FP)
  
  # precision <- TP/(TP+FP)
  # recall <- TP/(TP+FN)
  # 
  # f1 = (2*precision*recall)/(precision+recall)
  return(rbind(accuracy, sensitivity, specificity))
}


metrics_calc_preds_label <- function(label, preds){
  auc_roc <- roc(as.factor(label),as.numeric(preds), 
                 plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
  auc <- auc_roc$auc[1]
  # ci(test_auc_all_cross)
  # test performance 
  Youden_index <- auc_roc$sensitivities + auc_roc$specificities
  
  
  best_cutoff <-  auc_roc$thresholds[Youden_index==max(Youden_index)]
  print('best cutoff = ')
  print(best_cutoff)
  
  acc_sen_spe <- func_acc_sen_spe(label,preds,best_cutoff,1)
  auc_acc_sen_spe <- c(auc, acc_sen_spe)
  results <- round(auc_acc_sen_spe, 3)
  print(paste('AUC=', results[1],', ACC=',results[2],', SEN=',results[3],', SPE=',results[4]), sep = "")
}
  
  
draw_cor_plots <- function(cor_data, x_name, y_name){
    b <- ggplot(cor_data, aes(x = preds, y = labels))
    # Scatter plot with regression line
    b + geom_point()+
      geom_smooth(method = "lm", color = "black", fill = "lightgray") 
    
    # b + geom_point(shape = 17)+
    #   geom_smooth(method = "lm", color = "black", fill = "lightgray")
    # ggpubr::show_point_shapes()
    
    # Add regression line and confidence interval
    # Add correlation coefficient: stat_cor()
    ggscatter(cor_data, x = "preds", y = "labels",
              add = "reg.line", conf.int = TRUE,    
              add.params = list(fill = "lightgray"), xlab=x_name,ylab=y_name
              
    )+
      stat_cor(method = "spearman",label.x = 3, label.y = 15, cex = 5)
}


# determine the cutoff for LN meta
cutoff_meta <- function(pN01, p_patches_preds_max){
  # determine the cutoff for the LN meta
  auc_roc <- roc(as.factor(pN01),p_patches_preds_max,
                 plot=TRUE, print.thres=TRUE, print.auc=TRUE, smooth = FALSE)
  auc <- auc_roc$auc[1]
  Youden_index <- auc_roc$sensitivities + auc_roc$specificities
  best_cutoff <-  auc_roc$thresholds[Youden_index==max(Youden_index)]
  print(best_cutoff)
  return (best_cutoff)
}


LN_meta_num <- function(p_names, patches_names, patches_pred_label){
  # calculate the metastatic LN number
  p_pred_LN_meta_num <- c(1:length(p_names))*0
  
  for(ind in seq(length=length(p_names),from = 1,to=length(p_names))){
    p_name_temp <- p_names[ind]
    p_patches_pred_label <- patches_pred_label[which(patches_names==p_name_temp)]
    p_pred_LN_meta_num[ind] <- sum(p_patches_pred_label)
  }
  return(p_pred_LN_meta_num)
}


cN_stage <- function(p_pred_LN_meta_num){
  cN <- p_pred_LN_meta_num*0
  cN[which(p_pred_LN_meta_num>0)] <- 1
  cN[which(p_pred_LN_meta_num>3)] <- 2
  return(cN)
}



