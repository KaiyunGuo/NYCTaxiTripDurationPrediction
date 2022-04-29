library(lubridate)
library(geosphere)
library(tidyverse)
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

# # cluster on pickup location
# # WSS <- rep(0, 10)
# # for (k in 1:10) {
# #     # extract the total within-group sum of squared errors
# #     WSS[k] = kmeans(x = scale(full_dat[, 5:6]), centers = k, nstart = 100)$tot.withinss
# # }
# # 
# # ggplot(mapping = aes(x = 1:10, y = WSS)) +
# #     geom_line() + 
# #     geom_point() +
# #     geom_vline(xintercept = 4) +
# #     scale_x_discrete(name = "k", limits = factor(1:10)) +
# #     labs(title = "Elbow Method")
# # k = 3 or k = 5 is the best
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
# 


# standardize numeric cols: dist_hav, trip_duration
# columns useful after cleaning
#full_dat$dist_hav = scale(full_dat$dist_hav)
full_dat$log_trip_duration = log(full_dat$trip_duration)
#full_dat$passenger_count = factor(full_dat$passenger_count)

# separate train_dat into train and validation group
# train, validationfull_dat = full_dat[, !(colnames(full_dat) %in% col)]
#col = c("log_trip_duration", "passenger_count", "day", "hour", "minuteofday",
#        "dist_hav", "pickup_loc", "dropoff_loc", "weekend", "direction")
col = c("id", "pickup_date","pickup_time", "trip_duration",
        "minuteofday", "pickup_loc", "dropoff_loc", "weekday")
full_dat = full_dat[, !(colnames(full_dat) %in% col)]
idx = sample(1:nrow(train_dat), round(nrow(train_dat)*3/4))
train_dat = full_dat[1:30000,]
train1 = train_dat[idx,]
validation1 = train_dat[-idx,]
# test
test1 = full_dat[30001:40000,]
box = boxplot(train1$log_trip_duration)$stats
train1 = filter(train1, log_trip_duration > box[1], log_trip_duration < box[5])

rm(train_dat)

# Random Forest
# cross-validation on m
ctrl <- trainControl(method = "cv", number = 5)
rf.Grid = expand.grid(mtry = 2:13)
rf.cv.model <- train(log_trip_duration ~ ., data = train1,
                     method = "rf",
                     trControl = ctrl,
                     tuneGrid = rf.Grid)
rf.cv.model
rf.cv.model$bestTune
# mtry = 5
rf.pred = predict(rf.cv.model, validation1)
mean((validation1[,"log_trip_duration"] - rf.pred)^2)
## 0.219595

# predict log duration
rlt <- predict(rf.cv.model, test1)
#write to a submission file
outDat = data.frame(id = test_dat$id, trip_duration = exp(rlt))
write.csv(outDat, "outlierRF.csv", row.names = F)



# package randomForest
rfModel = randomForest(log_trip_duration ~ ., data = train1, 
                       importance = TRUE, 
                       ntree=500, mtry = 5)
rfModel
varImpPlot(rfModel)
plot(rfModel)
# predict on validation set
rf.pred1 = predict(rfModel, validation1)
mean((validation1[,"log_trip_duration"] - rf.pred1)^2)
# predict log duration
rlt1 <- predict(rfModel, test1)
#write to a submission file
outDat = data.frame(id = test_dat$id, trip_duration = exp(rlt))
write.csv(outDat, "outlierRFfunc.csv", row.names = F)





