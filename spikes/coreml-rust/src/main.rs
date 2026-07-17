#![deny(unsafe_op_in_unsafe_fn)]

use std::{env, path::Path, slice};

use objc2::{
    AnyThread,
    rc::autoreleasepool,
    runtime::{AnyObject, ProtocolObject},
};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

const DEFAULT_REPETITIONS: usize = 20;

fn main() {
    let model_path = env::var("CUTTLEDOC_COREML_MODEL").unwrap_or_else(|_| {
        eprintln!("set CUTTLEDOC_COREML_MODEL to a compiled .mlmodelc directory");
        std::process::exit(2);
    });
    let repetitions = env::var("CUTTLEDOC_COREML_REPETITIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_REPETITIONS);

    if !Path::new(&model_path).is_dir() {
        eprintln!("CoreML model directory does not exist: {model_path}");
        std::process::exit(2);
    }

    for run in 1..=repetitions {
        let speech_probability = autoreleasepool(|_| run_once(&model_path));
        match speech_probability {
            Ok(speech_probability) => println!("run={run} vad_output={speech_probability:.7}"),
            Err(error) => {
                eprintln!("run={run} failed: {error}");
                std::process::exit(1);
            }
        }
    }
}

fn run_once(model_path: &str) -> Result<f32, String> {
    let path = NSString::from_str(model_path);
    let url = NSURL::fileURLWithPath(&path);
    let configuration = unsafe { MLModelConfiguration::new() };
    unsafe { configuration.setComputeUnits(MLComputeUnits::All) };

    let model = unsafe {
        MLModel::modelWithContentsOfURL_configuration_error(&url, &configuration)
            .map_err(coreml_error)?
    };

    let audio = zeroed_array(&[1, 576])?;
    let hidden = zeroed_array(&[1, 128])?;
    let cell = zeroed_array(&[1, 128])?;
    let values = unsafe {
        [
            MLFeatureValue::featureValueWithMultiArray(&audio),
            MLFeatureValue::featureValueWithMultiArray(&hidden),
            MLFeatureValue::featureValueWithMultiArray(&cell),
        ]
    };
    let features: objc2::rc::Retained<NSDictionary<NSString, AnyObject>> =
        NSDictionary::from_slices(
            &[
                &*NSString::from_str("audio_input"),
                &*NSString::from_str("hidden_state"),
                &*NSString::from_str("cell_state"),
            ],
            &[values[0].as_ref(), values[1].as_ref(), values[2].as_ref()],
        );
    let input = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &features,
        )
        .map_err(coreml_error)?
    };
    let input: &ProtocolObject<dyn MLFeatureProvider> = ProtocolObject::from_ref(&*input);
    let output = unsafe {
        model
            .predictionFromFeatures_error(input)
            .map_err(coreml_error)?
    };
    let probability = unsafe {
        output
            .featureValueForName(&NSString::from_str("vad_output"))
            .ok_or_else(|| "model output is missing vad_output".to_owned())?
            .multiArrayValue()
            .ok_or_else(|| "vad_output is not an MLMultiArray".to_owned())?
    };

    read_single_f32(&probability)
}

fn zeroed_array(shape: &[usize]) -> Result<objc2::rc::Retained<MLMultiArray>, String> {
    let shape_numbers = shape
        .iter()
        .map(|dimension| NSNumber::new_usize(*dimension))
        .collect::<Vec<_>>();
    let shape = NSArray::from_retained_slice(&shape_numbers);
    let array = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape,
            MLMultiArrayDataType::Float32,
        )
        .map_err(coreml_error)?
    };
    let count = unsafe { array.count() } as usize;
    #[allow(deprecated)]
    let data = unsafe { array.dataPointer().cast::<f32>() };
    let values = unsafe { slice::from_raw_parts_mut(data.as_ptr(), count) };
    values.fill(0.0);
    Ok(array)
}

fn read_single_f32(array: &MLMultiArray) -> Result<f32, String> {
    let count = unsafe { array.count() } as usize;
    if count != 1 {
        return Err(format!("expected one VAD output scalar, received {count}"));
    }
    #[allow(deprecated)]
    let data = unsafe { array.dataPointer().cast::<f32>() };
    Ok(unsafe { *data.as_ptr() })
}

fn coreml_error(error: objc2::rc::Retained<objc2_foundation::NSError>) -> String {
    format!("{error:?}")
}
