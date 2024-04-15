import mindspore as ms
import numpy as np
import time


class LossMonitor(ms.Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        per_print_times (int): How many steps to print once loss. During sink mode, it will print loss in the
                               nearest step. Default: 1.

    Raises:
        ValueError: If per_print_times is not an integer or less than zero.
    """

    def __init__(self, accum_iter=1, per_print_times=100, ifeval=True, log=None):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.log = log
        self.losses = []
        self.eval = ifeval
        self.accum_iter = accum_iter

        # 额外添加的计时模块
        self.time = time.time()

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()

        if isinstance(cb_params.net_outputs, tuple):
            # for Ascend
            loss = cb_params.net_outputs[0].asnumpy()
        else:
            # for CPU/GPU
            loss = cb_params.net_outputs.asnumpy()
        loss *= self.accum_iter
        self.losses.append(loss)

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))

        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            costtime = time.time() - self.time
            self.time = time.time()
            self.log.info("epoch: %s step: %s, loss is %s,time cost is %.2fs" % (
            cb_params.cur_epoch_num, cur_step_in_epoch, loss, costtime))

    # pylint: disable=unused-argument
    def on_train_epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def on_train_epoch_end(self, run_context):
        callback_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / callback_params.batch_num

        if self.eval:
            eval_results = callback_params.eval_results
            print(f"Epoch time: {epoch_mseconds:5.3f} ms, "
                  f"per step time: {per_step_mseconds:5.3f} ms, "
                  f"avg loss: {np.mean(self.losses):5.3f}, "
                  f"metric: {eval_results}", flush=True)
        else:
            print(f"Epoch time: {epoch_mseconds:5.3f} ms, "
                  f"per step time: {per_step_mseconds:5.3f} ms, "
                  f"avg loss: {np.mean(self.losses):5.3f}", flush=True)
            


class StopAtStep(ms.Callback):
    """
    Start profiling base on step.

    Args:
        start_step (int): The start step number.
        stop_step (int): The stop step number.
    """
    def __init__(self, start_step, stop_step):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = ms.Profiler(start_profile=False, output_path='./profiler_data')

    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
            self.profiler.analyse()
