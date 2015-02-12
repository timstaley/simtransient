from scipy import stats
from datetime import timedelta


def generate_poisson_event_timestamps(inverse_rate, start, end):
    period = end-start
    # expected = rate * period = period / inverse_rate
    mean_per_period = period.total_seconds() / inverse_rate.total_seconds()
    n_events = stats.poisson.rvs(mu=mean_per_period, size=1)[0]
    event_times = []
    for _ in xrange(n_events):
        event_times.append(
            start +
               timedelta(seconds = stats.uniform.rvs()*period.total_seconds()))
    return sorted(event_times)


def generate_survey_alert_timestamp(survey_waveband, survey_faint_limit,
                                    survey_cadence, transient):
    pass