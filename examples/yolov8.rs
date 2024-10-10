#![allow(clippy::manual_retain)]

use std::path::Path;

use image::{imageops::FilterType, GenericImageView};
use ndarray::{s, Array, Axis};
use ort::{inputs, ROCmExecutionProvider, Session, 
    SessionOutputs,CUDAExecutionProvider,CPUExecutionProvider,Error};

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}
impl BoundingBox {
    fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        let x1 = box1.x1.max(box2.x1);
        let y1 = box1.y1.max(box2.y1);
        let x2 = box1.x2.min(box2.x2);
        let y2 = box1.y2.min(box2.y2);

        let width = (x2 - x1).max(0.0);
        let height = (y2 - y1).max(0.0);

        width * height
    }

    fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        let intersection_area = Self::intersection(box1, box2);
        let box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
        let box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

        box1_area + box2_area - intersection_area
    }
}

    

#[derive(Debug, Clone, Copy)]
struct ResultBox {
    bounding_box: BoundingBox,
    label: &'static str,
    prob: f32,
}

struct YOLOv8_DETECTIOR {
    model: Session,
    img_size: (u32, u32),
     boxes:  Vec<ResultBox>,
   

}

impl YOLOv8_DETECTIOR {
    pub fn new(env_name: &str, model_name: &str, imgze : (u32, u32)) -> Result<Self, ort::Error> {
        let model = match env_name {
            "rocm" => {
                ort::init()
                    .with_execution_providers([ROCmExecutionProvider::default().build()])
                    .commit()?;
                Session::builder()?.commit_from_file(model_name)?
            }
            "cuda" => {
                ort::init()
                    .with_execution_providers([CUDAExecutionProvider::default().build()])
                    .commit()?;
                Session::builder()?.commit_from_file(model_name)?
            }
            "cpu" => {
                ort::init()
                    .with_execution_providers([CPUExecutionProvider::default().build()])
                    .commit()?;
                Session::builder()?.commit_from_file(model_name)?
            }
            _ => return Err(Error::new("Invalid environment name")),
        };

        Ok(Self { model ,
            img_size: imgze,
            boxes: Vec::new(),
        
        })
    }
  
    fn load_images(&self, image_bytes: &[u8]) -> Array<f32, ndarray::Dim<[usize; 4]>> {
        let original_img = image::load_from_memory(image_bytes).unwrap();
        let (img_width, img_height) = (original_img.width(), original_img.height());
        let img = original_img.resize_exact(self.img_size.0, self.img_size.1, FilterType::CatmullRom);
        let mut input = Array::zeros((1, 3, 640, 640));
        for pixel in img.pixels() {
            let x = pixel.0 as usize;
            let y = pixel.1 as usize;
            let [r, g, b, _] = pixel.2.0;
            input[[0, 0, y, x]] = (r as f32) / 255.;
            input[[0, 1, y, x]] = (g as f32) / 255.;
            input[[0, 2, y, x]] = (b as f32) / 255.;
        }
        input
    }

    pub fn infer(&mut self, input: Array<f32, ndarray::Dim<[usize; 4]>>) -> Result<(), ort::Error> {
        let (img_width, img_height) = self.img_size;
        let outputs: SessionOutputs = self.model.run(inputs!["images" => input.view()]?)?;
        let output = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();

        self.boxes.clear(); // 清空之前的检测结果

        let output = output.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();
            if prob < 0.5 {
                continue;
            }
            let label = YOLOV8_CLASS_LABELS[class_id];
            let xc = row[0] / 640. * (img_width as f32);
            let yc = row[1] / 640. * (img_height as f32);
            let w = row[2] / 640. * (img_width as f32);
            let h = row[3] / 640. * (img_height as f32);
            self.boxes.push(ResultBox {
                bounding_box: BoundingBox {
                    x1: xc - w / 2.,
                    y1: yc - h / 2.,
                    x2: xc + w / 2.,
                    y2: yc + h / 2.,
                },
                label,
                prob,
            });
        }
        Ok(())
    }
    pub fn nms(&mut self, iou_threshold: f32) -> Vec<ResultBox> {
        self.boxes.sort_by(|box1, box2| box2.prob.total_cmp(&box1.prob));
        let mut result = Vec::new();

        while !self.boxes.is_empty() {
            result.push(self.boxes[0].clone());
            self.boxes = self.boxes
                .iter()
                .filter(|box1| {
                    let last_box = result.last().unwrap();
                    BoundingBox::intersection(&last_box.bounding_box, &box1.bounding_box) / BoundingBox::union(&last_box.bounding_box, &box1.bounding_box) < iou_threshold
                })
                .cloned()
                .collect::<Vec<ResultBox>>();
        }

        result
    }
      
    

}



#[rustfmt::skip]
const YOLOV8_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

fn main() {
    tracing_subscriber::fmt::init();
    let mut detector = YOLOv8_DETECTIOR::new("rocm", "yolov8s.onnx", (640, 640)).unwrap();
    let image_bytes = std::fs::read(Path::new("data/baseball.jpg")).unwrap();
    let input = detector.load_images(&image_bytes);
    detector.infer(input).unwrap();
    let result = detector.nms(0.5);
    print!("{:?}", result);
   
}