package(default_visibility = ["//visibility:public"])

py_binary(
    name = "train",
    srcs = ["main.py"],
    data = ["//:deepmind_lab.so"],
    main = "main.py"
)

py_binary(
    name = "display",
    srcs = ["display.py"],
    data = ["//:deepmind_lab.so"],
    main = "display.py"
)

py_binary(
    name = "visualize",
    srcs = ["visualize.py"],
    data = ["//:deepmind_lab.so"],
    main = "visualize.py"
)

py_test(
    name = "test",
    srcs = ["test.py"],
    main = "unreal/test.py",
    deps = [":train"],
)
