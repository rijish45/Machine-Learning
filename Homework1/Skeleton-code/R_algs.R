library(kernlab)
library(rpart)
library(ada)
library(RWeka)
library(randomForest)

n <- nrow(t_all)
d <- ncol(t_all)
set.seed(100)
sample_indices <- sample(n,n)
b1 <- round((n/3),digits=0)
b2 <- 2*b1
fold1 <- sample_indices[1:b1]
fold2 <- sample_indices[(b1+1):b2]
fold3 <- sample_indices[(b2+1):n]
t1 <- t_all[fold1,]
t2 <- t_all[fold2,]
t3 <- t_all[fold3,]
nrow(t1)
nrow(t2)
nrow(t3)
y1 <- y_all[fold1]
y2 <- y_all[fold2]
y3 <- y_all[fold3]

colnames_t <- "V1"
for (j in 2:d) colnames_t <- c(colnames_t, paste("V",j,sep=""))
formula_lm <- "y~V1"
for (j in 2:d) formula_lm <- paste(formula_lm,"+V",j,sep="")

num_algs <- 6
results <- matrix(seq(from=0,to=0,length.out=(6*num_algs)), nrow=6, ncol=num_algs)
colnames(results) <- c("LR", "SVM", "CART", "C45", "RF", "ADA")
rownames(results) <- c("Train12", "Test3", "Train13", "Test2", "Train23", "Test1")

num_leaves <- matrix(nrow=1,ncol=8)
colnames(num_leaves) <- c("CART_1", "C4.5_1", "CART_2", "C4.5_2", "CART_3", "C4.5_3", "CART_Avg", "C4.5_Avg")
for (f in 1:3){
  if (f==1){
    t <- rbind(t1,t2)
    y <- c(y1,y2)
    t_test <- t3
    y_test <- y3
  }
  if (f==2){
    t <- rbind(t1,t3)
    y <- c(y1,y3)
    t_test <- t2
    y_test <- y2
  }
  if (f==3){
    t <- rbind(t2,t3)
    y <- c(y2,y3)
    t_test <- t1
    y_test <- y1
  }

  colnames(t) <- colnames_t
  colnames(t_test) <- colnames_t
  data_train <- cbind(t,y)
  colnames(data_train) <- c(colnames_t,"y")

  # logistic regression
  glm_model <- glm(formula_lm, family=binomial(link="logit"), data=as.data.frame(data_train))
  y_hat <- round(predict(glm_model, newdata=as.data.frame(t), type="response"))
  results[(2*f-1),1] <- (sum(1-abs(y-y_hat)))/length(y)
  y_hat <- round(predict(glm_model, newdata=as.data.frame(t_test), type="response"))
  results[(2*f),1] <- (sum(1-abs(y_test-y_hat)))/length(y_test)

  # SVM
  set.seed(100)
  svm_model <- ksvm(x=as.matrix(t), y=as.factor(y))
  y_hat <- predict(svm_model, newdata=t, type="response")
  results[(2*f-1),2] <- (sum(1-abs(y-(as.numeric(y_hat)-1))))/length(y)
  y_hat <- predict(svm_model, newdata=t_test, type="response")
  results[(2*f),2] <- (sum(1-abs(y_test-(as.numeric(y_hat)-1))))/length(y_test)
 
  # CART
  cart_model <- rpart(formula_lm, data=as.data.frame(data_train))
  y_hat <- round(predict(cart_model, newdata=as.data.frame(t)))
  results[(2*f-1),3] <- (sum(1-abs(y-y_hat)))/length(y)
  y_hat <- round(predict(cart_model, newdata=as.data.frame(t_test)))
  results[(2*f),3] <- (sum(1-abs(y_test-y_hat)))/length(y_test)
  num_leaves[1,(2*f-1)] <- length(which(cart_model$frame[,"var"]=="<leaf>"))

  # C4.5
  data_train_fac <- as.data.frame(data_train)
  colnames(data_train_fac) <- c(colnames_t, "y")
  data_train_fac[,"y"] <- as.factor(data_train_fac[,"y"])
  c45_model <- J48(formula_lm, data=as.data.frame(data_train_fac))
  y_hat <- predict(c45_model, newdata=as.data.frame(t), type="class")
  results[(2*f-1),4] <- (sum(1-abs(y-(as.numeric(y_hat)-1))))/length(y)
  y_hat <- predict(c45_model, newdata=as.data.frame(t_test), type="class")
  results[(2*f),4] <- (sum(1-abs(y_test-(as.numeric(y_hat)-1))))/length(y_test)
  num_leaves[1,(2*f)] <- c45_model$classifier$measureNumRules()

  # RandomForests
  rf_model <- randomForest(x=t, y=as.factor(y))
  y_hat <- predict(rf_model, newdata=as.data.frame(t), type="class")
  results[(2*f-1),5] <- (sum(1-abs(y-(as.numeric(y_hat)-1))))/length(y)
  y_hat <- predict(rf_model, newdata=as.data.frame(t_test), type="class")
  results[(2*f),5] <- (sum(1-abs(y_test-(as.numeric(y_hat)-1))))/length(y_test)

  # Adaboost
  set.seed(100)
  boost_model <- ada(x=t, y=y)
  y_hat <- predict(boost_model, newdata=as.data.frame(t))
  results[(2*f-1),6] <- (sum(1-abs(y-(as.numeric(y_hat)-1))))/length(y)
  y_hat <- predict(boost_model, newdata=as.data.frame(t_test))
  results[(2*f),6] <- (sum(1-abs(y_test-(as.numeric(y_hat)-1))))/length(y_test)
}

results_avg <- matrix(nrow=2, ncol=num_algs)
colnames(results_avg) <- colnames(results)
rownames(results_avg) <- c("Train","Test")
results_sd <- matrix(nrow=2, ncol=num_algs)
colnames(results_sd) <- colnames(results)
rownames(results_sd) <- c("Train","Test")
results_avg[1,] <- colMeans(results[c(1,3,5),])
results_avg[2,] <- colMeans(results[c(2,4,6),])
results_sd[1,] <- apply(results[c(1,3,5),],2,sd)
results_sd[2,] <- apply(results[c(2,4,6),],2,sd)
results_avg
results_sd

results

num_leaves[1,7] <- mean(c(num_leaves[1,1],num_leaves[1,3],num_leaves[1,5]))
num_leaves[1,8] <- mean(c(num_leaves[1,2],num_leaves[1,4],num_leaves[1,6]))
num_leaves


