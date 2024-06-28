import numpy as np

from .. import utilities as util

def remove_unimportant_overloads(l_overloads):
    return list(filter(lambda o: not o.is_unimportant(), l_overloads))

class OverloadEvent:
    """Storage of metrics associated with a single event of __potentially__ overloaded line

    Attributes:
        dt_start (datetime) : Starting time of event
        dt_end (datetime) : Ending time of event
        duration_h (int) : Duration of event in hours
        fl_spike (float) : Highest observed overload during the event
        fl_surplus_energy_MWh (float) : Total energy above the allowed power limit
        fl_rms_load (float) : RMS power demand during event
        percentage_overload (float) : Percentage of RMS power above limit
        fl_ramping (float) : Approximate ramping speed up to peak overload
    """

    def __init__(self, ts_overload_event, fl_power_limit) -> None:
        """Calculate metrics associated with an event

        Args:
            ts_overload_event (timeseries) : Load during event
            fl_power_limit (float) : Power limit of line in question during the event
        """

        # Time-metrics
        self.dt_start = ts_overload_event[0,0]
        self.dt_end = ts_overload_event[-1,0]
        dt_duration = self.dt_end - self.dt_start
        self.duration_h = util.duration_to_hours(dt_duration)

        # Power-metrics
        fl_peak_surplus = np.max(ts_overload_event[:, 1]) - fl_power_limit
        self.fl_spike = max(
            fl_peak_surplus, 0
        )  # Negative overload (surplus capacity) not considered

        fl_energy = 0
        fl_rms_load = 0
        for i in range(1, len(ts_overload_event)):  # Max Riemann-sum
            fl_max_load_surplus = (
                max(ts_overload_event[i - 1, 1], ts_overload_event[i, 1])
                - fl_power_limit
            )
            fl_max_overload = max(
                fl_max_load_surplus, 0
            )  # Negative overload (surplus capacity) not considered

            # Can be simplified if hour-requirement is assumed
            dt_dur = (ts_overload_event[i, 0] - ts_overload_event[i - 1, 0])
            fl_dur = util.duration_to_hours(dt_dur)

            fl_energy += fl_max_overload * fl_dur
            fl_rms_load += ts_overload_event[i - 1, 1] * fl_dur
        self.fl_surplus_energy_MWh = fl_energy

        fl_rms_load = fl_rms_load / self.duration_h
        self.fl_rms_load = fl_rms_load
        self.percentage_overload = 100 * fl_rms_load / fl_power_limit

        # TODO: Ramping should really be calculated from the first timestamp before peak where only load increases are observed
        # As in, the power could actually decrease from the time of dt_start, while ramping should be calculated from
        # the first time it only is increasing.
        spike_dur = ts_overload_event[np.argmax(ts_overload_event[:,1]),0] - self.dt_start
        spike_dur_h = util.duration_to_hours(spike_dur)
        self.fl_ramping = (np.max(ts_overload_event[:,1]) - fl_power_limit)/spike_dur_h if spike_dur_h else -1

    def __str__(self):
        return (
            "Overload Event with properties:\n"
            + "Start              :     "
            + str(self.dt_start)
            + "   \n"
            + "End                :     "
            + str(self.dt_end)
            + "   \n"
            + "Duration           :     "
            + str(self.duration_h)
            + "   \n"
            + "------------\n"
            + "Spike over limit   :     "
            + str(self.fl_spike)
            + "   kW\n"
            + "RMS load           :     "
            + str(self.fl_rms_load)
            + "   kW\n"
            + "Energy over limit  :     "
            + str(self.fl_surplus_energy_MWh)
            + "   kWh\n"
            + "% Overload         :     "
            + str(self.percentage_overload)
            + "   %\n"
            + "Ramping            :     "
            + str(self.fl_ramping)
            + "   kW/h\n"
        )

    def is_unimportant(self):
        # TODO: Other ways of characterizing "unimportant load"
        if self.duration_h == 1:
            b_short = True
        else:
            b_short = False
        return b_short


class FlexibilityNeed:
    """
    A timeseries may have many occurences of overloads.
    A flexibility need is defined as a meta-metric over all the overload-events.
    """

    def __init__(self, l_overloads) -> None:
        self.l_overloads = l_overloads
        self.str_flex_category = "" 

        l_recovery_times = []
        num_overloads = len(l_overloads)
        for i in range(num_overloads):
            if i != num_overloads - 1:  # cannot find recovery-time for last event
                dt_recovery_time = l_overloads[i + 1].dt_start - l_overloads[i].dt_end
                l_recovery_times.append(dt_recovery_time)
        l_recovery_times.append(util.undef_timedelta())   # Last overload-event has undefined recovery-time
        self.l_recovery_times = l_recovery_times

        self.fl_avg_frequency = np.average([1 / util.duration_to_hours(t) for t in self.l_recovery_times])
        self.fl_avg_spike = np.average([o.fl_spike for o in l_overloads])

    def extract_arrays(self):
        arrs = {}
        l_overloads = self.l_overloads

        arrs["spike"] = np.array([o.fl_spike for o in l_overloads])
        arrs["energy"] = np.array([o.fl_surplus_energy_MWh for o in l_overloads])
        arrs["duration"] = np.array([o.duration_h for o in l_overloads])
        arrs["season"] = np.array([util.datetime_to_season(o.dt_start) for o in l_overloads])
        arrs["month"] = np.array([o.dt_start.month for o in l_overloads])
        arrs["recovery"] = np.array([util.duration_to_hours(t) for t in self.l_recovery_times])
        arrs["ramping"] = np.array([o.fl_ramping for o in l_overloads])

        return arrs

def metric_annotation(metric_name):
    # TODO: Change to match case when python 3.10 ubiquitous
    ann = ""
    if metric_name == "spike":
        ann = "Spike [kW]"
    elif metric_name == "energy":
        ann = "Energy [kWh]"
    elif metric_name == "duration":
        ann = "Duration [h]"
    elif metric_name == "season":
        ann = "Season"
    elif metric_name == "month":
        ann = "Month"
    elif metric_name == "recovery":
        ann = "Recovery-time [h]"
    elif metric_name == "ramping":
        ann = "Ramping [kW/h]"
    return ann
