# Return period

A return period, also known as a recurrence interval or repeat interval, is an average time or an estimated average time between events such as earthquakes, floods, landslides, or river discharge flows to occur.

The reciprocal value of return period is called the frequency of occurrence.

It is a statistical measurement typically based on historic data over an extended period, and is used usually for risk analysis. Examples include deciding whether a project should be allowed to go forward in a zone of a certain risk or designing structures to withstand events with a certain return period. The following analysis assumes that the probability of the event occurring does not vary over time and is independent of past events.

m is the rank of observed occurrences when arranged in descending order

For floods, the event may be measured in terms of m3/s or height; for storm surges, in terms of the height of the surge, and similarly for other events.  This is Weibull's Formula.

Return period as the reciprocal of expected frequency

The theoretical return period between occurrences is the inverse of the average frequency of occurrence. For example, a 10-year flood has a 1/10 = 0.1 or 10% chance of being exceeded in any one year and a 50-year flood has a 0.02 or 2% chance of being exceeded in any one year.

This does not mean that a 100-year flood will happen regularly every 100 years, or only once in 100 years.  Despite the connotations of the name "return period". In any given 100-year period, a 100-year event may occur once, twice, more, or not at all, and each outcome has a probability that can be computed as below.

Also, the estimated return period below is a statistic: it is computed from a set of data (the observations), as distinct from the theoretical value in an idealized distribution. One does not actually know that a certain or greater magnitude happens with 1% probability, only that it has been observed exactly once in 100 years.

That distinction is significant because there are few observations of rare events: for instance, if observations go back 400 years, the most extreme event (a 400-year event by the statistical definition) may later be classed, on longer observation, as a 200-year event (if a comparable event immediately occurs) or a 500-year event (if no comparable event occurs for a further 100 years).

Further, one cannot determine the size of a 1000-year event based on such records alone but instead must use a statistical model to predict the magnitude of such an (unobserved) event. Even if the historic return interval is a lot less than 1000 years, if there are a number of less-severe events of a similar nature recorded, the use of such a model is likely to provide useful information to help estimate the future return interval.

One would like to be able to interpret the return period in probabilistic models. The most logical interpretation for this is to take the return period as the counting rate in a Poisson distribution since it is the expectation value of the rate of occurrences. An alternative interpretation is to take it as the probability for a yearly Bernoulli trial in the binomial distribution. That is disfavoured because each year does not represent an independent Bernoulli trial but is an arbitrary measure of time. This question is mainly academic as the results obtained will be similar under both the Poisson and binomial interpretations.

The probability mass function of the Poisson distribution is

{\displaystyle P(r;t)={(\mu t)^{r} \over r!}e^{-\mu t}={(t/T)^{r} \over r!}e^{-t/T}}

is the number of occurrences the probability is calculated for,

The probability of no-occurrence can be obtained simply considering the case for

Consequently, the probability of exceedance (i.e. the probability of an event "stronger" than the event with return period

to occur at least once within the time period of interest) is

{\displaystyle P(t>0;t)=1-P(t=0;t)=1-e^{-\mu t}=1-e^{-t/T}}

, the probability of exceedance within an interval equal to the return period (i.e.

) is independent from the return period and it is equal to

. This means, for example, that there is a 63.2% probability of a flood larger than the 50-year return flood to occur within any period of 50 year.

) then the probability of exactly one occurrence in ten years is

{\displaystyle {\begin{aligned}P(r;t)&={\frac {(\mu t)^{r}}{r!}}e^{-\mu t}\\[6pt]P(r=1;t=10)&={\frac {(10/243)^{1}}{1!}}e^{-10/243}\approx 3.95\%\end{aligned}}}

), the probability of a given number r of events of a return period

{\displaystyle P(X=r)={n \choose r}\mu ^{r}(1-\mu )^{n-r}.}

This is valid only if the probability of more than one occurrence per unit time

is zero. Often that is a close approximation, in which case the probabilities yielded by this formula hold approximately.

{\displaystyle n\rightarrow \infty ,\mu \rightarrow 0}

{\displaystyle {\frac {n!}{(n-r)!r!}}\mu ^{r}(1-\mu )^{n-r}\rightarrow e^{-\lambda }{\frac {\lambda ^{r}}{r!}}.}

m is the number of recorded occurrences of the event being considered

Given that the return period of an event is 100 years,

So the probability that such an event occurs exactly once in 10 successive years is:

{\displaystyle {\begin{aligned}P(X=1)&={\binom {10}{1}}\times 0.01^{1}\times 0.99^{9}\\[4pt]&\approx 10\times 0.01\times 0.914\\[4pt]&\approx 0.0914\end{aligned}}}

Return period is useful for risk analysis (such as natural, inherent, or hydrologic risk of failure). When dealing with structure design expectations, the return period is useful in calculating the riskiness of the structure.

The probability of at least one event that exceeds design limits during the expected life of the structure is the complement of the probability that no events occur which exceed design limits.

{\displaystyle {\overline {R}}=1-\left(1-{1 \over T}\right)^{n}=1-(1-P(X\geq x_{T}))^{n}}

is the expression for the probability of the occurrence of the event in question in a year;