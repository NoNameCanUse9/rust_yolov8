#![allow(clippy::manual_retain)]

use std::option;
use std::path::Path;
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc,core};
use image::{imageops::FilterType, GenericImageView};
use ndarray::{s, Array, Axis};
use ort::{inputs, ROCmExecutionProvider, Session, 
    SessionOutputs,CUDAExecutionProvider,CPUExecutionProvider,Error};
use tracing_subscriber::fmt::writer::MutexGuardWriter;

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

struct Yolov8Detectior {
    model: Session,
    img_size: (i32, i32),
    boxes:  Vec<ResultBox>,
    infer_mat: Mat,
    scale: Option<f32>,
    pad_w: Option<i32>,
    pad_h: Option<i32>,
   

}

impl Yolov8Detectior {
    pub fn new(env_name: &str, model_name: &str, imgze : (i32, i32)) -> Result<Self, ort::Error> {
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
            infer_mat: Mat::default(),
            scale: None, // 初始化为 None
            pad_w: None, // 初始化为 None
            pad_h: None, // 初始化为 None
        })
    }
  
    // fn load_images(&self, image_bytes: &[u8]) -> Array<f32, ndarray::Dim<[usize; 4]>> {
    //     let original_img = image::load_from_memory(image_bytes).unwrap();
    //     let (img_width, img_height) = (original_img.width(), original_img.height());
    //     let img = original_img.resize_exact(self.img_size.0, self.img_size.1, FilterType::CatmullRom);
    //     let mut input = Array::zeros((1, 3, 640, 640));
    //     for pixel in img.pixels() {
    //         let x = pixel.0 as usize;
    //         let y = pixel.1 as usize;
    //         let [r, g, b, _] = pixel.2.0;
    //         input[[0, 0, y, x]] = (r as f32) / 255.;
    //         input[[0, 1, y, x]] = (g as f32) / 255.;
    //         input[[0, 2, y, x]] = (b as f32) / 255.;
    //     }
    //     input
    // }
    fn letterbox(&mut self,) -> Result<Mat, opencv::Error> {
        let mat = self.infer_mat.clone();
        let (h, w) = (mat.rows(), mat.cols());
        let scale = (self.img_size.0 as f32 / w as f32).min(self.img_size.1 as f32 / h as f32);
       
        let new_w = (w as f32 * scale).round() as i32;
        let new_h = (h as f32 * scale).round() as i32;
        let pad_w = (self.img_size.0 - new_w) / 2;
        let pad_h = (self.img_size.1 - new_h) / 2;
        let mut resized_img = Mat::default();
        opencv::imgproc::resize(&mat, &mut resized_img, core::Size { width: new_w, height: new_h }, 0.0, 0.0, imgproc::INTER_LINEAR)?;
        let mut padded_img = Mat::default();
        opencv::core::copy_make_border(&resized_img, &mut padded_img, pad_h, pad_h, pad_w, pad_w, opencv::core::BORDER_CONSTANT, opencv::core::Scalar::all(255.0))?;
        self.scale = Some(scale);
        //保存图片
        self.pad_w = Some(pad_w); // 设置 self.pad_w
    self.pad_h = Some(pad_h); // 设置 self.pad_h
        Ok(padded_img)
    }
    
   pub fn load_image_to_mat(& mut self,image_path: &str) -> Result<(), opencv::Error> {
        // 加载图片
        self.infer_mat= imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
        
        
        // 创建一个临时变量来存储颜色转换后的图像
    
   

        // 交换临时变量和原始变量的内容
   
        Ok(())
         
    }
        
    pub fn infer(&mut self, mat:& mut Mat ) -> Result<(), ort::Error> {
    
        let (img_width, img_height) = self.img_size;
        let input_array = ndarray::Array4::<f32>::from_shape_fn((1, 3, 640, 640), |(_, c, y, x)| {
            mat.at_2d::<core::Vec3b>(y as i32, x as i32).unwrap()[c] as f32 / 255.0
        });
        let outputs: SessionOutputs = self.model.run(inputs!["images" => input_array.view()]?)?;
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
        self.boxes.sort_by(|box1: &ResultBox, box2: &ResultBox| box2.prob.total_cmp(&box1.prob));
        let mut result: Vec<ResultBox> = Vec::new();
    
        while !self.boxes.is_empty() {
            let box1 = self.boxes.remove(0); // 移除第一个边界框
            result.push(box1.clone());
    
            self.boxes = self.boxes
                .iter()
                .filter(|&box2| {
                    BoundingBox::intersection(&box1.bounding_box, &box2.bounding_box) / BoundingBox::union(&box1.bounding_box, &box2.bounding_box) < iou_threshold
                })
                .cloned()
                .collect::<Vec<ResultBox>>();
        }
    
        // 转换回原图的坐标点
        let scale = self.scale.unwrap_or(1.0); // 使用 self.scale
        let pad_w = self.pad_w.unwrap_or(0); // 使用 self.pad_w
        let pad_h = self.pad_h.unwrap_or(0); // 使用 self.pad_h
    
        for box1 in &mut result {
            box1.bounding_box.x1 = (box1.bounding_box.x1 - pad_w as f32) / scale;
            box1.bounding_box.y1 = (box1.bounding_box.y1 - pad_h as f32) / scale;
            box1.bounding_box.x2 = (box1.bounding_box.x2 - pad_w as f32) / scale;
            box1.bounding_box.y2 = (box1.bounding_box.y2 - pad_h as f32) / scale;
        }
    
        println!("{:?}", result);
        result
    }
    pub fn draw_boxes(&mut self , boxes:&Vec<ResultBox>) -> Result<(), opencv::Error> {
        for box1 in boxes {
          
            let x1 = box1.bounding_box.x1 as i32;
            let y1 = box1.bounding_box.y1 as i32;
            let x2 = box1.bounding_box.x2 as i32;
            let y2 = box1.bounding_box.y2 as i32;
            let color = core::Scalar::new(0.0, 255.0, 0.0, 0.0);
            let font = imgproc::FONT_HERSHEY_SIMPLEX;
            let font_scale = 0.5;
            let thickness = 1;
            let mut baseline = 0;
            let text = format!("{} {:.2}", box1.label, box1.prob);
            let text_size = imgproc::get_text_size(&text, font, font_scale, thickness, &mut baseline)?;
            let text_origin = core::Point::new(x1, y1 + text_size.height);
            imgproc::rectangle(&mut self.infer_mat, core::Rect::new(x1, y1, x2 - x1, y2 - y1), color, 2, 8, 0)?;
            imgproc::put_text(&mut self.infer_mat, &text, text_origin, font, font_scale, color, thickness, 8, false)?;
        }
        imgcodecs::imwrite("data/baseball_result.jpg", &self.infer_mat, &core::Vector::new())?;
        Ok(())
    }
    pub fn inference (&mut self,image_path: &str ) {
        self.load_image_to_mat(image_path).unwrap();
        let mut mat = self.letterbox().unwrap();
        self.infer(& mut mat).unwrap();
        let result: Vec<ResultBox> = self.nms(0.5);
        {
            self.draw_boxes(&result).unwrap();}
        {
        //imgcodecs::imwrite("data/baseball_result.jpg", &mut self.infer_mat, &core::Vector::new()).unwrap();
        }

        
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
    let mut detector: Yolov8Detectior = Yolov8Detectior::new("rocm", "yolov8s.onnx", (640, 640)).unwrap();
    // let image_bytes = std::fs::read(Path::new("data/baseball.jpg")).unwrap();
    //detector.load_image_to_mat("data/baseball.jpg").unwrap();
    
   
    detector.inference("data/baseball.jpg");
   
}