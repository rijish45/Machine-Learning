% compare_mlAlgorithms.m
%
% This MATLAB script serves as an experimental platform for comparing
% different machine learning algorithms.
%

close all
clc

%% Data Loading

load creditCard
p = 0.8; % proportion of the dataset for training

% ... convert the dataset to matrices
[X, Y, Xtest, Ytest] = ds2matrices(creditCard, p);

% 2D data visualization by PCA
[~,score,~] = pca(X); 
gscatter(score(:,1), score(:,2),Y); 
hold off;


%% Generalized Linear Model (Logistic Regression)

glmModel = fitglm(X, Y, 'Distribution', 'binomial', 'Link', 'logit');
Yscores = predict(glmModel, Xtest); % these are the posterior probabilities
                                    % of class 1 for the test data

% ... compute the standard ROC curve and the AUROC
[Xglm, Yglm, Tglm, AUCglm] = perfcurve(Ytest, Yscores, 'true');

%% Support Vector Machine (SVM)

svmModel = fitcsvm(X, Y, 'Standardize', true, 'KernelFunction', 'rbf');
svmModel = fitPosterior(svmModel);
[~, Yscores] = predict(svmModel, Xtest);

% ... compute the standard ROC curve and the AUROC
[Xsvm, Ysvm, Tsvm, AUCsvm] = perfcurve(Ytest, Yscores(:, 2), 'true');

%% Classification Tree (CART)

ctreeModel = fitctree(X, Y);
[~, Yscores, ~, ~] = predict(ctreeModel, Xtest);

% ... compute the standard ROC curve and the AUROC
[Xcart, Ycart, Tcart, AUCcart] = perfcurve(Ytest, Yscores(:, 2), 'true');

%% Random Forest (RF)

rfModel = fitensemble(X, Y, 'Bag', 100, 'Tree', 'Type', 'Classification');
[~, Yscores] = predict(rfModel, Xtest);

% ... compute the standard ROC curve and the AUROC
[Xrf, Yrf, Trf, AUCrf] = perfcurve(Ytest, Yscores(:, 2), 'true');

%% Boosted Trees

btModel = fitensemble(X, Y, 'AdaBoostM1', 100, 'Tree');
[~, Yscores] = predict(btModel, Xtest);

% ... compute the standard ROC curve and the AUROC
[Xbt, Ybt, Tbt, AUCbt] = perfcurve(Ytest, sigmf(Yscores(:, 2), [1 0]), ...
                                   'true');

%% ROC Curves

plot(Xglm, Yglm)
hold on
plot(Xsvm, Ysvm)
plot(Xcart, Ycart)
plot(Xrf, Yrf)
plot(Xbt, Ybt)
legend('Logistic Regression', 'Support Vector Machine', 'CART', ...
       'Random Forest', 'Boosted Trees')
xlabel('false positive rate');
ylabel('true positive rate');
title('ROC Curves for Classification Algorithms')
hold off

%% AUROC

fprintf('AUROC for\n');
fprintf('Logistic Regression: %f\n', AUCglm);
fprintf('Support Vector Machine: %f\n', AUCsvm);
fprintf('CART: %f\n', AUCcart);
fprintf('Random Forest: %f\n', AUCrf);
fprintf('Boosted Trees: %f\n', AUCbt);

return

%% Program Log
%  first created: Chaofan Chen, August 21, 2016
%  last modified:
%  Duke University, Machine Learning, Fall 2016