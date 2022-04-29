library(lubridate)
library(geosphere)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(caret)
library(randomForest)
library(xgboost)
library(Matrix)
library(MASS)
library(factoextra)
library(gbm)
set.seed(3)

# read data
train_dat <- read.csv("W22P1_train.csv", header = TRUE)
test_dat <- read.csv("W22P1_test.csv", header = TRUE)

test_dat = mutate(test_dat, trip_duration = rep(0, nrow(test_dat)), .before = 1)	
full_dat = rbind(train_dat, test_dat)


# extract day from pickup_date
full_dat$day = day(full_dat$pickup_date)
# extract hour and mintue from pickup_time
full_dat$hour = hour(hms(full_dat$pickup_time))
#Surprisingly pickup_min was much more important than pickup_hour???
full_dat$minute = minute(hms(full_dat$pickup_time))
full_dat$minuteofday = full_dat$hour * 60 + full_dat$minute
# Day of week, Sun == 1, Sat == 7
full_dat$weekday = wday(as.Date(full_dat$pickup_date))
# Weekend?
full_dat$weekend =  (full_dat$weekday == 1 | full_dat$weekday == 7) + 0


# compute distance using Haversine
full_dat$dist_hav <- distHaversine(full_dat[,c("pickup_longitude","pickup_latitude")],
                               full_dat[,c("dropoff_longitude","dropoff_latitude")])

# Haversine distance
# geo_dist <- function(lon1, lat1, lon2, lat2){
#     lon1 = lon1 * pi / 180
#     lat1 = lat1 * pi / 180
#     lon2 = lon2 * pi / 180
#     lat2 = lat2 * pi / 180
#     dlon = lon2 - lon1 
#     dlat = lat2 - lat1 
#     a = (sin(dlat/2))^2 + cos(lat1) * cos(lat2) * (sin(dlon/2))^2 
#     c = 2 * atan2( sqrt(a), sqrt(1-a) ) 
#     return(6378137 * c)  # distance in meters
# }

# driving direction
full_dat$direction <- bearing(full_dat[,c("pickup_longitude","pickup_latitude")],
                                   full_dat[,c("dropoff_longitude","dropoff_latitude")])

# cluster on pickup location
# WSS <- rep(0, 10)
# for (k in 1:10) {
#     # extract the total within-group sum of squared errors
#     WSS[k] = kmeans(x = scale(full_dat[, 5:6]), centers = k, nstart = 100)$tot.withinss
# }
# 
# ggplot(mapping = aes(x = 1:10, y = WSS)) +
#     geom_line() + 
#     geom_point() +
#     geom_vline(xintercept = 4) +
#     scale_x_discrete(name = "k", limits = factor(1:10)) +
#     labs(title = "Elbow Method")
# k = 3 or k = 5 is the best

# pickup_kmeans <- kmeans(x = scale(full_dat[, 5:6]), centers = 5, nstart = 500)
# full_dat$pickup_loc  = pickup_kmeans$cluster
# 
# # # cluster on dropoff location
# # WSS <- rep(0, 10)
# # for (k in 1:10) {
# #     # extract the total within-group sum of squared errors
# #     WSS[k] = kmeans(x = scale(full_dat[, 7:8]), centers = k, nstart = 100)$tot.withinss
# # }
# # 
# # ggplot(mapping = aes(x = 1:10, y = WSS)) +
# #     geom_line() + 
# #     geom_point() +
# #     geom_vline(xintercept = 4) +
# #     scale_x_discrete(name = "k", limits = factor(1:10)) +
# #     labs(title = "Elbow Method")
# # k = 3 or k = 4 is the best
# dropoff_kmeans <- kmeans(x = scale(full_dat[, 7:8]), centers = 4, nstart = 500)
# full_dat$dropoff_loc  = dropoff_kmeans$cluster



# standardize numeric cols: dist_hav, trip_duration
# columns useful after cleaning
#full_dat$dist_hav = scale(full_dat$dist_hav)
full_dat$trip_duration = log(full_dat$trip_duration)
#full_dat$passenger_count = full_dat$passenger_count

# separate train_dat into train and validation group
# train, validation
#col = c("log_trip_duration", "passenger_count", "day", "hour", "minuteofday",
#        "dist_hav", "pickup_loc", "dropoff_loc", "weekend", "direction")
col = c("id", "pickup_date","pickup_time","day","minuteofday","weekday")
full_dat = full_dat[, !(colnames(full_dat) %in% col)]
idx = sample(1:nrow(train_dat), round(nrow(train_dat)*3/4))
train_dat = full_dat[1:30000,]
train1 = train_dat[idx,]
validation1 = train_dat[-idx,]
# test
test1 = full_dat[30001:40000,]

# remove outliers in train data
box = boxplot(train1$trip_duration)$stats
train1 = filter(train1, trip_duration > box[1], trip_duration < box[5])

# XGboost
# nrounds (# Boosting Iterations)
# max_depth (Max Tree Depth)
# eta (Shrinkage)
# gamma (Minimum Loss Reduction)
# colsample_bytree (Subsample Ratio of Columns)
# min_child_weight (Minimum Sum of Instance Weight)
# subsample (Subsample Percentage)
input_x = as.matrix(train1[,-6])
input_y = train1[,6]

# helper function for the plots
tuneplot <- function(x, probs = .90) {
    ggplot(x) +
        coord_cartesian(ylim = c(min(x$results$RMSE),
                                 quantile(x$results$RMSE, probs = probs))) +
        theme_bw()
}


nrounds =1000
tune_grid <- expand.grid(
    nrounds = seq(from = 200, to = nrounds, by = 50),
    eta = c(0.025, 0.05, 0.1, 0.3),
    max_depth = c(6,8,10,12),
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
)

tune_control <- caret::trainControl(
    method = "cv",
    number = 3, # n folds 
    verboseIter = FALSE, # no training log
    allowParallel = TRUE # FALSE for reproducible results 
)

xgb_tune <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid,
    method = "xgbTree",
    verbose = TRUE
)


tuneplot(xgb_tune)
xgb_tune$bestTune
# nrounds max_depth   eta gamma colsample_bytree
# 900         6 0.025     0                1
# min_child_weight subsample
# 15                1         1
xgb_tune
min(xgb_tune$results$RMSE)
#  0.3402988

# fixed eta = 0.025, max_depth = c(6 +- 1)
tune_grid2 <- expand.grid(
    nrounds = seq(from = 50, to = nrounds, by = 50),
    eta = xgb_tune$bestTune$eta,
    max_depth = c(5:7),
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = c(1, 2, 3),
    subsample = 1
)

xgb_tune2 <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid2,
    method = "xgbTree",
    verbose = TRUE
)

tuneplot(xgb_tune2)
xgb_tune2$bestTune
# nrounds max_depth   eta gamma colsample_bytree
# 550         7 0.025     0                1
# min_child_weight subsample
# 3         1
min(xgb_tune2$results$RMSE)
## 0.3392327

# fixed eta = 0.025
# max_depth = 7, min_child_weight = 3
tune_grid3 <- expand.grid(
    nrounds = seq(from = 50, to = nrounds, by = 50),
    eta = xgb_tune$bestTune$eta,
    max_depth = xgb_tune2$bestTune$max_depth,
    gamma = 0,
    colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
    min_child_weight = xgb_tune2$bestTune$min_child_weight,
    subsample = c(0.5, 0.75, 1.0)
)

xgb_tune3 <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid3,
    method = "xgbTree",
    verbose = TRUE
)

tuneplot(xgb_tune3, probs = .95)
xgb_tune3$bestTune
# nrounds max_depth   eta gamma colsample_bytree
# 650         7 0.025     0              0.8
# min_child_weight subsample
# 3      0.75
min(xgb_tune3$results$RMSE)
## 0.3372224


# column sampling = 0.8 and row sampling = 0.75
tune_grid4 <- expand.grid(
    nrounds = seq(from = 50, to = nrounds, by = 50),
    eta = xgb_tune$bestTune$eta,
    max_depth = xgb_tune2$bestTune$max_depth,
    gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
    colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
    min_child_weight = xgb_tune2$bestTune$min_child_weight,
    subsample = xgb_tune3$bestTune$subsample
)

xgb_tune4 <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid4,
    method = "xgbTree",
    verbose = TRUE
)

tuneplot(xgb_tune4)
xgb_tune4$bestTune
# nrounds max_depth   eta gamma colsample_bytree
# 700         7 0.025     0              0.8
# min_child_weight subsample
#   3      0.75
min(xgb_tune4$results$RMSE)
## 0.3381932


# gamma = 0
tune_grid5 <- expand.grid(
    nrounds = seq(from = 100, to = 5000, by = 100),
    eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
    max_depth = xgb_tune2$bestTune$max_depth,
    gamma = xgb_tune4$bestTune$gamma,
    colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
    min_child_weight = xgb_tune2$bestTune$min_child_weight,
    subsample = xgb_tune3$bestTune$subsample
)

xgb_tune5 <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = tune_grid5,
    method = "xgbTree",
    verbose = TRUE
)

tuneplot(xgb_tune5)
xgb_tune5$bestTune
# nrounds max_depth  eta gamma colsample_bytree
# 2000         7 0.01     0              0.8
# min_child_weight subsample
# 3      0.75
min(xgb_tune5$results$RMSE)
# 0.335198

xgb.pred = predict(xgb_tune5, validation1)
mean((xgb.pred - validation1$trip_duration)^2)
#1000 0.2137874
# 0.2222259
rlt <- predict(xgb_tune5, test1)
#write to a submission file
outDat = data.frame(id = test_dat$id, trip_duration = exp(rlt))
write.csv(outDat, "xgboost_tuning.csv", row.names = F)
###NEW BEST


(final_grid <- expand.grid(
    nrounds = xgb_tune5$bestTune$nrounds,
    eta = xgb_tune5$bestTune$eta,
    max_depth = xgb_tune5$bestTune$max_depth,
    gamma = xgb_tune5$bestTune$gamma,
    colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
    min_child_weight = xgb_tune5$bestTune$min_child_weight,
    subsample = xgb_tune5$bestTune$subsample
))