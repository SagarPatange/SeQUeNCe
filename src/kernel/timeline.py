"""Definition of main Timeline class.

This module defines the Timeline class, which provides an interface for the simulation kernel and drives event execution.
All entities are required to have an attached timeline for simulation.
"""

from _thread import start_new_thread
from math import inf
from sys import stdout
from time import time_ns, sleep
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .event import Event

from .eventlist import EventList


class Timeline:
    """Class for a simulation timeline.

    Timeline holds entities, which are configured before the simulation.
    Before the start of simulation, the timeline must initialize all controlled entities.
    The initialization of entities may schedule events.
    The timeline pushes these events to its event list.
    The timeline starts simulation by popping the top event in the event list repeatedly.
    The time of popped event becomes current simulation time of the timeline.
    The process of popped event is executed.
    The simulation stops if the timestamp on popped event is equal or larger than the stop time, or if the eventlist is empty.

    To monitor the progress of simulation, the Timeline.show_progress attribute can be modified to show/hide a progress bar.

    Attributes:
        events (EventList): the event list of timeline.
        entities (List[Entity]): the entity list of timeline used for initialization.
        time (int): current simulation time (picoseconds).
        stop_time (int): the stop (simulation) time of the simulation.
        is_running (bool): records if the simulation has stopped executing events.
        show_progress (bool): show/hide the progress bar of simulation.
        logflag (bool): determines if timeline should log events.
    """

    def __init__(self, stop_time=inf):
        """Constructor for timeline.

        Args:
            stop_time (int): stop time (in ps) of simulation (default inf).
        """

        self.events = EventList()
        self.entities = []
        self.time = 0
        self.stop_time = stop_time
        self.event_counter = 0
        self.is_running = False

        self.show_progress = False
        self.logflag = False
        self._logger = None

    def now(self) -> int:
        """Returns current simulation time."""

        return self.time

    def schedule(self, event: "Event") -> None:
        """Method to schedule an event."""

        self.event_counter += 1
        return self.events.push(event)

    def init(self) -> None:
        """Method to initialize all simulated entities.

        Also sets timeline logger.
        """

        if self.logflag:
            from ..utils.log import new_sequence_logger
            self._logger = new_sequence_logger(__name__)

        for entity in self.entities:
            entity.init()

    def run(self) -> None:
        """Main simulation method.

        The `run` method begins simulation of events.
        Events are continuously popped and executed, until the simulation time limit is reached or events are exhausted.
        A progress bar may also be displayed, if the `show_progress` flag is set.
        """

        self.is_running = True

        if self.show_progress:
            self.progress_bar()

        # log = {}
        while len(self.events) > 0:
            event = self.events.pop()
            if event.time >= self.stop_time:
                self.schedule(event)
                break
            assert self.time <= event.time, "invalid event time for process scheduled on " + str(event.process.owner)
            self.time = event.time
            # if not event.process.activation in log:
            #     log[event.process.activation] = 0
            # log[event.process.activation]+=1
            event.process.run()

        # print('number of event', self.event_counter)
        # print('log:',log)

        self.is_running = False

    def stop(self) -> None:
        """Method to stop simulation."""

        self.stop_time = self.now()

    def remove_event(self, event: "Event") -> None:
        self.events.remove(event)

    def update_event_time(self, event: "Event", time: int) -> None:
        """Method to change execution time of an event.

        Args:
            event (Event): event to reschedule.
            time (int): new simulation time (should be >= current time).
        """

        self.events.remove(event)
        event.time = time
        self.schedule(event)

    def seed(self, seed: int) -> None:
        """Sets random seed for simulation."""

        from numpy import random
        random.seed(seed)

    def log(self, caller: any, level: int, message: str):
        """Method to access timeline log.

        Before using, must:

        1. Set `logfile` flag in utils.logs module.
        2. Call timeline `init` method to initialize logger.

        Args:
            caller (any): object calling the log method.
            level (int): log level (defined from default `logging` module).
            message (str): message to log.
        """

        if self.logflag:
            message = " ".join([str(self.now()), message])
            self._logger.log(level, message)

    def progress_bar(self):
        """Method to draw progress bar.

        Progress bar will display the execution time of simulation, as well as the current simulation time.
        """

        def print_time():
            start_time = time_ns()
            while self.is_running:
                exe_time = self.ns_to_human_time(time_ns() - start_time)
                sim_time = self.ns_to_human_time(self.time / 1e3)
                if self.stop_time == float('inf'):
                    stop_time = 'NaN'
                else:
                    stop_time = self.ns_to_human_time(self.stop_time / 1e3)
                process_bar = f'\rexecution time: {exe_time};     simulation time: {sim_time} / {stop_time}'
                print(f'{process_bar}', end="\r")
                stdout.flush()
                sleep(3)

        start_new_thread(print_time, ())

    def ns_to_human_time(self, nanosec: int) -> str:
        if nanosec >= 1e6:
            ms = nanosec / 1e6
            nanosec = nanosec % 1e6
            if ms >= 1e3:
                second = ms / 1e3
                if second >= 60:
                    minute = second // 60
                    second = second % 60
                    if minute >= 60:
                        hour = minute // 60
                        minute = minute % 60
                        return '%d hour: %d min: %.2f sec' % (hour, minute, second)
                    return '%d min: %.2f sec' % (minute, second)
                return '%.2f sec' % (second)
            return "%d ms, %.2f ns" % (ms, nanosec)
        return '%d ns' % nanosec
