using System;
using System.IO;
using Microsoft.ML;

namespace CareerPrediction.Trainer
{
    public class Test
    {
        public static void RunPredictionTest()
        {
            // 1. Khởi tạo MLContext độc lập dành riêng cho việc chạy thử nghiệm (Dự đoán)
            MLContext mlContext = new MLContext();

            // 2. Định nghĩa đường dẫn trỏ thẳng tới file cấu trúc model .zip đã lưu trước đó
            string modelPath = Path.Combine(Environment.CurrentDirectory, "CareerModel.zip");

            // Kiểm tra xem file model có tồn tại thực tế hay không
            if (!File.Exists(modelPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[LỖI] Không tìm thấy file bộ não AI tại đường dẫn: {modelPath}");
                Console.WriteLine("Mẹo: Hãy chạy luồng huấn luyện ở file Program.cs trước để tạo ra file .zip này.");
                Console.ResetColor();
                return;
            }

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n========================================================");
            Console.WriteLine("    CHƯƠNG TRÌNH ĐÁNH GIÁ VÀ CHẠY THỬ NGHIỆM MÔ HÌNH AI ");
            Console.WriteLine("========================================================");
            Console.ResetColor();

            // 3. Nạp mô hình đã train từ file .zip vào bộ nhớ RAM
            ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

            // 4. Khởi tạo Prediction Engine (Bộ cơ chế dự đoán chuyên biệt của ML.NET)
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            bool keepRunning = true;
            while (keepRunning)
            {
                // Cấu hình hiển thị font tiếng Việt cho Console nhập dữ liệu
                Console.OutputEncoding = System.Text.Encoding.UTF8;
                Console.InputEncoding = System.Text.Encoding.UTF8;

                Console.WriteLine("\n--------------------------------------------------------");

                // Thu thập thông tin kiểm thử từ bàn phím
                Console.Write("✏️ Nhập mã tổ hợp xét tuyển (Ví dụ: A00, A01, D01, C00): ");
                string maToHop = Console.ReadLine()?.Trim().ToUpper();

                Console.Write("✏️ Nhập tổng điểm của tổ hợp thi (Ví dụ: 24.50): ");
                string diemStr = Console.ReadLine()?.Trim();
                // Rửa dữ liệu: chuyển đổi linh hoạt giữa dấu phẩy và dấu chấm đề phòng người dùng gõ nhầm
                if (!float.TryParse(diemStr?.Replace(',', '.'), out float diemToHop))
                {
                    diemToHop = 20.0f; // Điểm mặc định nếu nhập sai định dạng số
                }

                Console.Write("✏️ Nhập nhóm tính cách ưu tú (Ví dụ: Investigative, Realistic, Social, Enterprising): ");
                string nhomTinhCach = Console.ReadLine()?.Trim();

                // 5. Đóng gói thông tin thô thành Object dữ liệu đầu vào chuẩn ML.NET
                var sampleInput = new ModelInput
                {
                    MaToHop = maToHop,
                    DiemToHop = diemToHop,
                    NhomTinhCach = nhomTinhCach
                };

                // 6. Ép AI thực thi lệnh suy luận logic và đưa ra dự đoán
                var prediction = predictionEngine.Predict(sampleInput);

                // 7. Xuất kết quả phán quyết của AI ra màn hình
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"👉 [AI PHÁN QUYẾT]: Với dữ liệu trên, học sinh phù hợp nhất với:");
                Console.WriteLine($"   🎯 NHÓM NGÀNH: {prediction.NhomNganhDuDoan}");
                Console.ResetColor();

                // Hỏi người dùng có muốn tiếp tục test bộ dữ liệu khác hay không
                Console.Write("\n🔄 Bạn có muốn tiếp tục nhập dữ liệu test khác không? (y/n): ");
                string choice = Console.ReadLine()?.Trim().ToLower();
                keepRunning = (choice == "y" || choice == "yes");
            }

            Console.WriteLine("\nĐã đóng chương trình Test. Tạm biệt!");
        }
    }
}