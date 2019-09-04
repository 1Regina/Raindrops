Using data scrapped from NEA website for Newton, I narrowed down from 2009 to 2019 data where there were gaps to 2014 to 2019 where all the parameters are recorded daily.

Selected initially temperature as the target and performed log functions for feature engineering on winds under the Sunbeam file.

Subsequently the target was changed to rainfall prediction. Raindrop is the file for source code for this project henceforth.

Applied Regression models - OLS, Lasso, Ridge (no ElasticNet) at different alphas and Polynomial to find the best R^2.

However, given that the target is not normally distributed, none of these regression models works. This was reflected via the error scattered plot, histograms of bin after sqroot and log on target and lastly the box-cox transformation. 

In summary, regression is not suitable when the target is not normally-distributed or more strictly, not conditionally-normally distributed ie where even its SSE is not normally distributed.