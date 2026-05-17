using System;
using System.IO;
using CareerPrediction.Trainer;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

class Program
{
    static void Main(string[] args)
    {
    //    // 1. Khởi tạo MLContext (Bộ não điều khiển của ML.NET)
    //    MLContext mlContext = new MLContext(seed: 42);

    //    // Đóng gói đường dẫn file dữ liệu (Hãy thay bằng đường dẫn thực tế của bạn)
    //    string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "dataset.csv");

    //    Console.WriteLine("--- BƯỚC 1: ĐANG TẢI DATASET ---");
    //    IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
    //        path: dataPath,
    //        hasHeader: true,
    //        separatorChar: ',');

    //    // Chia dữ liệu: 80% để Train (Học), 20% để Test (Đánh giá thuật toán)
    //    var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 42);
    //    var trainData = split.TrainSet;
    //    var testData = split.TestSet;

    //    Console.WriteLine("--- BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU (FEATURE ENGINEERING) ---");
    //    // ML.NET chỉ hiểu số thô, nên ta phải chuyển chữ (MaToHop, NhomTinhCach) thành Vector số (OneHotEncoding)
    //    // Cột nhãn NhomNganh cũng phải chuyển thành dạng Key số để phân loại đa lớp
    //    var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(ModelInput.NhomNganh))
    //        .Append(mlContext.Transforms.Categorical.OneHotEncoding("EncodedToHop", nameof(ModelInput.MaToHop)))
    //        .Append(mlContext.Transforms.Categorical.OneHotEncoding("EncodedTinhCach", nameof(ModelInput.NhomTinhCach)))
    //        .Append(mlContext.Transforms.Concatenate("Features", "EncodedToHop", "EncodedTinhCach", nameof(ModelInput.DiemToHop)))
    //        .AppendCacheCheckpoint(mlContext); // Lưu bộ nhớ cache để tăng tốc độ train

    //    Console.WriteLine("--- BƯỚC 3: THỬ NGHIỆM VÀ ĐÁNH GIÁ 3 THUẬT TOÁN ---\n");

    //    var sdcaOptions = new SdcaMaximumEntropyMulticlassTrainer.Options
    //    {
    //        LabelColumnName = "Label",
    //        FeatureColumnName = "Features",
    //        L2Regularization = 0.01f,
    //        MaximumNumberOfIterations = 200
    //    };
    //    // SỬA DÒNG NÀY: Thêm MapKeyToValue vào cuối pipeline
    //    var sdcaPipeline = dataProcessPipeline
    //        .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(sdcaOptions))
    //        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    //    EvaluateModel(mlContext, sdcaPipeline, trainData, testData, "SdcaMaximumEntropy");

    //    // ==========================================
    //    // THUẬT TOÁN 2: LightGBM Multiclass
    //    // ==========================================
    //    var fastTreeOptions = new LightGbmMulticlassTrainer.Options
    //    {
    //        LabelColumnName = "Label",
    //        FeatureColumnName = "Features",
    //        NumberOfLeaves = 20,
    //        NumberOfIterations = 100,
    //        LearningRate = 0.2f
    //    };
    //    // SỬA DÒNG NÀY: Thêm MapKeyToValue vào cuối pipeline
    //    var lgbmPipeline = dataProcessPipeline
    //        .Append(mlContext.MulticlassClassification.Trainers.LightGbm(fastTreeOptions))
    //        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    //    var bestModel = EvaluateModel(mlContext, lgbmPipeline, trainData, testData, "LightGBM Multiclass");

    //    // 4. LƯU MÔ HÌNH SAU KHI FINE-TUNE ĐẠT KẾT QUẢ TỐT NHẤT
    //    Console.WriteLine("\n--- BƯỚC 4: ĐÓNG GÓI MODEL XUẤT RA FILE .ZIP ---");
    //    string modelPath = Path.Combine(Environment.CurrentDirectory, "CareerModel.zip");
    //    mlContext.Model.Save(bestModel, trainData.Schema, modelPath);
    //    Console.WriteLine($"Đã lưu mô hình tối ưu nhất tại: {modelPath}");
    //    Console.ReadKey();
    //}

    //// Hàm phụ trách Huấn luyện và In số liệu đánh giá (Accuracy, Log-loss)
    //// Sử dụng ITransformer làm kiểu trả về và IEstimator<ITransformer> làm tham số đầu vào
    //static ITransformer EvaluateModel(
    //    MLContext mlContext,
    //    IEstimator<ITransformer> pipeline,
    //    IDataView trainData,
    //    IDataView testData,
    //    string modelName)
    //{
    //    Console.WriteLine($"=== Đang train thuật toán: {modelName} ===");

    //    // Thực hiện huấn luyện (Học) - Trả về một ITransformer chung
    //    var trainedModel = pipeline.Fit(trainData);

    //    // Chạy kiểm tra trên tập dữ liệu Test (Thi thử)
    //    var predictions = trainedModel.Transform(testData);
    //    var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

    //    // In các thông số chứng minh với giáo viên
    //    Console.WriteLine($"[KẾT QUẢ {modelName.ToUpper()}]:");
    //    Console.WriteLine($" - Micro-Accuracy (Độ chính xác tổng thể): {metrics.MicroAccuracy * 100:F2}%");
    //    Console.WriteLine($" - Macro-Accuracy (Độ chính xác trung bình lớp): {metrics.MacroAccuracy * 100:F2}%");
    //    Console.WriteLine($" - Log-Loss (Sai số toán học): {metrics.LogLoss:F4}");
    //    Console.WriteLine("--------------------------------------------------\n");


    //    return trainedModel;
        // Chỉ để lại duy nhất dòng này trong Program.cs
        CareerPrediction.Trainer.Test.RunPredictionTest();
    }


}