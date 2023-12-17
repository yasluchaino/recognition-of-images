using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork1
{

    public class StudentNetwork : BaseNetwork
    {
        public class Layer
        {
            //входные веса
            public double[] input = null;
            //входыне первоначальные данные
            public static double[] data = null;
            //предыдущий слой
            public Layer prev = null;
            //матрица весов
            public double[,] weights = null;

            //генератор для инициализации весов
            public static Random randGenerator = new Random();

            public Layer(Layer pr, int l)
            {
                input = new double[l];
                //доп нейрон со значением 1 или -1
                input[l - 1] = 1;
                prev = pr;
            }

          
            public static double[] MultVecMatrix(double[] m1, double[,] m2)
            {  
                double[] res = new double[m2.GetLength(1)];
                for (int i = 0; i < m2.GetLength(1); i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < m1.Length - 1; j++)
                    {
                        sum += m1[j] * m2[j, i];
                    }

                    res[i] = sum;
                }
                return res;
            }

            
            // Уст входные данные образа
            
            public void SetData()
            {
                input = data;
            }

            
            // Уст входные данные
            
            public void SetData(double[] inp)
            {
                input = inp;
            }
            
            //Функция активации
            public double Sigmoid(double a)
            {
                return 1 / (1 + Math.Exp(-a));
            }

            //пересчёт функцию активации
            public double[] FuncActivate()
            {
                double[] temp = MultVecMatrix(input, weights);
                double[] matrix = new double[temp.Length + 1];
                for (int i = 0; i < temp.Length; i++)
                {
                    matrix[i] = Sigmoid(temp[i]);
                }
                matrix[matrix.Length - 1] = 1;
                return matrix;
            }
            
        }
        
        public List<Layer> layers = new List<Layer>();
        public double Speed = 0.1;
        public Stopwatch Stopwatch = new Stopwatch();

        //иниц нач знач весов
        private static double initMinWeight = -1;
        private static double initMaxWeight = 1;
        
        // Конструктор сети с указанием структуры (количество слоёв и нейронов в них)
        // structure - массив с указанием нейронов на каждом слое (включая сенсорный)
        public StudentNetwork(int[] structure)
        {
            Layer prev = null;
            for (int i = 0; i < structure.Length; i++)
            {
                Layer temp = new Layer(prev, structure[i] + 1);
                layers.Add(temp);
                //зададим рандомно веса
                if (prev != null)
                {
                    prev.weights = new double[prev.input.Length, temp.input.Length - 1];
                    for (int ii = 0; ii < prev.input.Length; ii++)
                    {
                        for (int j = 0; j < temp.input.Length - 1; j++)
                        {
                            prev.weights[ii, j] = initMinWeight + Layer.randGenerator.NextDouble() * (initMaxWeight - initMinWeight);
                        }
                    }
                }
                prev = temp;
            }
        }

        // Обратное распространение ошибки        
        public double[][] Backward(double[] SampleOutput)
        {
            int length = layers.Count;
            double[][] res = new double[layers.Count][];
            for (int i = 0; i < length; i++)
            {
                //без свободного
                res[i] = new double[layers[i].input.Length - 1];
            }

            for (int i = length - 1; i > 0; i--)
            {
                //если слой последний
                if (i == length - 1)
                {
                    for (int j = 0; j < layers[i].input.Length - 1; j++)
                    {
                        double yj = layers[i].input[j];
                        //высчитываем дельту
                        res[i][j] = yj * (1 - yj) * (SampleOutput[j] - yj);
                    }
                }

                else
                {
                    for (int j = 0; j < layers[i].input.Length - 1; j++)
                    {
                        double yj = layers[i].input[j];
                        double sum = 0.0;
                        for (int k = 0; k < res[i + 1].Length; k++)
                        {
                            sum += res[i + 1][k] * layers[i].weights[j, k];
                        }
                        //дельта
                        res[i][j] = yj * (1 - yj) * sum;
                    }
                }
            }

            return res;
        }
        
        
        // здесь заполняем вхолные веса каждого слоя при помощи FunActivate
        public void CreateNeuralNetwork()
        {
            for (int i = 0; i < layers.Count; i++)
            {
                if (i == 0)
                {
                    layers[i].SetData();
                }
                else
                {
                    layers[i].input = layers[i].prev.FuncActivate();
                }
            }
        }
        
        
        //то же самое но для вх данных
        public void CreateNeuralNetwork(double[] inp)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                if (i == 0)
                {
                    layers[i].SetData(inp);
                }
                else
                {
                    layers[i].input = layers[i].prev.FuncActivate();
                }
            }
        }
       
        // суммарная квадратичная ошибка сети
        //output это у нас выходные данные образа
        public double EstimatedError(double[] output)
        {
            double res = 0.0;
            for (int i = 0; i < output.Length; ++i)
                res += Math.Pow(layers.Last().input[i] - output[i], 2);
            return res / 2;
        }

        //проходимся по сети 
        public double Forward(Sample sample)
        {
            Layer.data = sample.input;
            CreateNeuralNetwork();
            double error = EstimatedError(sample.Output);
            double[][] deltas = Backward(sample.Output);
            //коррекция весов
            for (int i = 1; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i - 1].input.Length; j++)
                {
                    for (int k = 0; k < layers[i].input.Length - 1; k++)
                    {
                        layers[i - 1].weights[j, k] += Speed * deltas[i][k] * layers[i - 1].input[j];
                    }
                }
            }

            return error;
        }

        // Обучение сети одному образ      
        //возвращает количество итераций для достижения заданного уровня ошибки 
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            //колво итераций
            int cnt = 0;
            while (Forward(sample) > acceptableError)
            {
                cnt++;
            }
            return cnt;
        }
        
        /// <summary>
        /// Обучаем нейросеть на готовом датасете
        /// </summary>
        /// <param name="samplesSet">Сам датасет</param>
        /// <param name="epochsCount">Количество эпох для тренировки</param>
        /// <param name="acceptableError"></param>
        /// <param name="parallel"></param>
        /// <returns></returns>
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            //общее количество образов, которые мы рассмотрим
            int allSamples = epochsCount * samplesSet.Count;
            //число реально просмотренных образов
            int samplesLooked = 0;
            double error = 0.0;
            double meanError;
            Stopwatch.Restart();
            for (int e = 0; e < epochsCount; e++)
            {
                for (int i = 0; i < samplesSet.samples.Count; i++)
                {
                    var s = samplesSet.samples[i];
                    error += Forward(s);//вычисляем ошибку на каждом образце и корректируем веса сети
                    samplesLooked++;
                    if (i % 100 == 0)
                    {
                        OnTrainProgress(1.0 * samplesLooked / allSamples, error / (e * samplesSet.Count + i + 1),
                            Stopwatch.Elapsed);
                    }
                }

                meanError = error / ((e + 1) * samplesSet.Count + 1);
                if (meanError <= acceptableError)
                {
                    OnTrainProgress(1.0, meanError, Stopwatch.Elapsed);
                    return meanError;
                }
            }
            meanError = error / (epochsCount * samplesSet.Count + 1);
            OnTrainProgress(1.0, meanError, Stopwatch.Elapsed);
            return meanError / (epochsCount * samplesSet.Count);
        }
        
        protected override double[] Compute(double[] input)
        {
            CreateNeuralNetwork(input);
            return layers.Last().input.Take(layers.Last().input.Length - 1).ToArray();
        }
    }
}