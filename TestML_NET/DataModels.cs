using Microsoft.ML.Data;

namespace CareerPrediction.Trainer
{
    // Lớp định nghĩa dữ liệu đầu vào từ file CSV
    public class ModelInput
    {
        [LoadColumn(0)]
        public string MaToHop { get; set; }

        [LoadColumn(1)]
        public float DiemToHop { get; set; }

        [LoadColumn(2)]
        public string NhomTinhCach { get; set; }

        [LoadColumn(3)]
        public string NhomNganh { get; set; } // Đây là cột Label
    }

    // Lớp hứng kết quả dự đoán từ Model
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string NhomNganhDuDoan { get; set; }

        public float[] Score { get; set; } // Mảng xác suất tương ứng với từng nhóm ngành
    }
}