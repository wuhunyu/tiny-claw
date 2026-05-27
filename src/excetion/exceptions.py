class TinyClawException(Exception):
    code: str = "TINY_CLAW_ERROR"
    message: str = "tiny claw 错误"

    def __init__(
            self,
            code: str | None = None,
            message: str | None = None,
    ):
        self.code = code if code is not None else self.code
        self.message = message if message is not None else self.message
        super().__init__(self.message)


# ==============================参数校验类==============================
class InvalidParamException(TinyClawException):
    code: str = "INVALID_PARAM"
    message: str = "无效参数"


# ==============================网络请求类==============================
class RequestException(TinyClawException):
    code: str = "REQUEST_ERROR"
    message: str = "请求错误"


class ResponseException(TinyClawException):
    code: str = "RESPONSE_ERROR"
    message: str = "响应错误"


class ResponseBlankException(TinyClawException):
    code: str = "RESPONSE_BLANK"
    message: str = "响应为空"


class ResponseParseException(TinyClawException):
    code: str = "RESPONSE_PARSE_ERROR"
    message: str = "响应解析错误"


# ============================== tool 类==============================
class ToolInvokeException(TinyClawException):
    code: str = "TOOL_INVOKE"
    message: str = "tool 执行错误"


# ============================== path 类==============================
class PathNotFoundException(TinyClawException):
    code: str = "PATH_NOT_FOUND"
    message: str = "路径不存在"


class PathOutsideWorkspaceException(TinyClawException):
    code: str = "PATH_OUTSIDE_WORKSPACE"
    message: str = "路径超出工作空间"


# ============================== file 类==============================
class FileNotExistException(TinyClawException):
    code: str = "FILE_NOT_EXIST"
    message: str = "文件不存在"


class NotFileException(TinyClawException):
    code: str = "NOT_FILE"
    message: str = "不是文件"


class FilePermissionException(TinyClawException):
    code: str = "FILE_PERMISSION"
    message: str = "文件权限错误"


# ==============================其它类==============================
class MatchException(TinyClawException):
    code: str = "MATCH_ERROR"
    message: str = "匹配错误"
