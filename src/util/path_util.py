import logging
import pathlib

logger = logging.getLogger(__name__)


def absolute_path(work_dir: str, path: str) -> str:
    """
    获取操作路径的绝对路径
    :param work_dir: 工作区路径
    :param path: 操作路径
    :return: 操作路径的绝对路径
    """
    if not work_dir or not path:
        logger.info("请指定 work_dir && path")
        raise ValueError("请指定 work_dir && path")
    # 获取工作空间路径的绝对路径
    absolute_work_dir = pathlib.Path(work_dir).expanduser().resolve()
    input_path = pathlib.Path(path).expanduser()
    # 如果是相对路径
    if not input_path.is_absolute():
        input_path = absolute_work_dir / input_path
    target_path = input_path.resolve()
    # 操作目录是否在工作目录下
    if not target_path.is_relative_to(absolute_work_dir):
        logger.info(f"{path} 不在工作区 {work_dir} 下")
        raise IOError(f"{path} 不在工作区 {work_dir} 下")
    return str(target_path)
