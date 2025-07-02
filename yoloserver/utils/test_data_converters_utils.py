from yoloserver.utils.data_converters_utils import convert_data_to_yolo
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    input_dir = "raw/original_annotations"
    yolo_output_dir = "raw/yolo_staged_labels"
    annotation_format = "pascal_voc"
    # final_classes_order = None  # 自动模式
    # 或者指定类别顺序
    # final_classes_order = ['head', 'ordinary_clothes', 'person', 'reflective_vest', 'safety_helmet']

    print(f"开始测试中间层转换，输入目录: {input_dir}")
    try:
        classes = convert_data_to_yolo(
            input_dir=input_dir,
            annotation_format=annotation_format,
            final_classes_order=None,  # 或者用上面那一行
            yolo_output_dir=yolo_output_dir
        )
        print("类别列表：", classes)
        print("测试完成，请检查输出目录和日志。")
    except Exception as e:
        print(f"测试失败: {e}") 