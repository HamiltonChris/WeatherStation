#include <TensorFlowLite.h>
#include <TensorFlow/lite/micro/all_ops_resolver.h>
#include <TensorFlow/lite/micro/micro_interpreter.h>
#include <TensorFlow/lite/micro/micro_log.h>
#include <TensorFlow/lite/micro/system_setup.h>
#include <TensorFlow/lite/schema/schema_generated.h>

#include "model2.h"
#include "bme280.h"
#include <Wire.h>

void i2c_send(uint8_t address, uint8_t* data, uint8_t length);
void i2c_receive(uint8_t address, uint8_t* data, uint8_t length);
void debug_print(char* str);

const tflite::Model *tflu_model = nullptr;
tflite::MicroInterpreter *tflu_interpreter = nullptr;
TfLiteTensor *tflu_i_tensor = nullptr;
TfLiteTensor * tflu_o_tensor = nullptr;

float tflu_i_scale = 1.0f;
int32_t tflu_i_zero_point = 0;
float tflu_o_scale = 1.0f;
int32_t tflu_o_zero_point = 0;

constexpr int t_sz = 4096;
uint8_t tensor_arena[t_sz] __attribute__((aligned(16)));

tflite::AllOpsResolver tflu_ops_resolver;

bme280_t bme280;
uint32_t pressure = 0;
uint32_t humidity = 0;
int32_t temperature = 0;
char user_data[100];
int8_t status = 0;

constexpr int num_hours = 3;
int8_t t_vals[num_hours] = {0};
int8_t h_vals[num_hours] = {0};
int cur_idx = 0;

// Calculated Z-scores
constexpr float t_mean = 2.63218;
constexpr float t_std = 7.44124;
constexpr float h_mean = 89.59602;
constexpr float h_std = 14.62136;

constexpr int num_reads = 3;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  while (!Serial);

  Wire.begin();

	bme280.send = &i2c_send;
	bme280.receive = &i2c_receive;
  bme280.print = &debug_print;
	bme280.temperature_oversampling = OVERSAMPLING_x1;
	bme280.pressure_oversampling = OVERSAMPLING_x1;
	bme280.humidity_oversampling = OVERSAMPLING_x1;
	bme280.standby = SB_1000MS;
	bme280.filter = FILTER_OFF;
	bme280.mode = NORMAL;

  status = bme280_init(&bme280);

  if (status)
  {
    Serial.print("Error initializing driver\n");
  }
  else
  {
    Serial.print("Sensor Initialization complete\n");
  }

  tflu_model = tflite::GetModel(snow_model_tflite);
  static tflite::MicroInterpreter static_interpreter(tflu_model, tflu_ops_resolver, tensor_arena, t_sz);
  tflu_interpreter = &static_interpreter;

  tflu_interpreter->AllocateTensors();
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);

  const auto* i_quant = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);
  const auto* o_quant = reinterpret_cast<TfLiteAffineQuantization*>(tflu_o_tensor->quantization.params);

  tflu_i_scale = i_quant->scale->data[0];
  tflu_i_zero_point = i_quant->zero_point->data[0];
  tflu_o_scale = o_quant->scale->data[0];
  tflu_o_zero_point = o_quant->zero_point->data[0];
}

void loop() {
  float t = 0.0f;
  float h = 0.0f;

  if (!status)
  {
    // status = bme280_read_all(&bme280, &pressure, &humidity, &temperature);
    if (status)
    {
      Serial.println("Error with config structure.");
    }
    else
    {
      // sprintf((char*)user_data, "Pressure: %d.%02d Pa, Temperature: %d.%02d degrees C, Humidity %d.%03d%%\r\n",
      //   pressure >> 8, (pressure & 0xFF) * 100 >> 8, temperature / 100, temperature % 100,
      //   humidity >> 10, (humidity & 0x3FF) * 1000 >> 10);
        
      // Serial.print(user_data);
        
      for (int i = 0; i < num_reads; ++i)
      {
        bme280_read_all(&bme280, &pressure, &humidity, &temperature);
        t += (float)temperature / 100;
        h += (float)humidity / 1000;
        delay(1000);
      }
      t /= (float)num_reads;
      h /= (float)num_reads;

      sprintf((char*)user_data, "Temp: %f deg C Humidity: %f%%", t, h);
      Serial.println(user_data);

      t = (t - t_mean) / t_std;
      h = (h - h_mean) / h_std;

      t = (t / tflu_i_scale);
      t += (float)tflu_i_zero_point;
      h = (h / tflu_i_scale);
      h += (float)tflu_i_zero_point;

      t_vals[cur_idx] = t;
      h_vals[cur_idx] = h;

      cur_idx = (cur_idx + 1) % num_hours;

      // Prepare input features
      int32_t idx0 = cur_idx;
      int32_t idx1 = (cur_idx - 1 + num_hours) % num_hours;
      int32_t idx2 = (cur_idx - 2 + num_hours) % num_hours;
      tflu_i_tensor->data.int8[0] = t_vals[idx2];
      tflu_i_tensor->data.int8[1] = t_vals[idx1];
      tflu_i_tensor->data.int8[2] = t_vals[idx0];
      tflu_i_tensor->data.int8[3] = h_vals[idx2];
      tflu_i_tensor->data.int8[4] = h_vals[idx1];
      tflu_i_tensor->data.int8[5] = h_vals[idx0];

      tflu_interpreter->Invoke();

      float out_int8 = tflu_o_tensor->data.int8[0];
      float out_f = (out_int8 - tflu_o_zero_point);
      out_f *= tflu_o_scale;

      if (out_f > 0.5)
      {
        Serial.println("Yes, it snows");
        Serial.println(out_f);
      }
      else
      {
        Serial.println("No, it doesn't snow");
      }

      delay(1000); // should be hours in production
    }
  }
  
}

// i2c callback function for writes
void i2c_send(uint8_t address, uint8_t* data, uint8_t length)
{
  Wire.beginTransmission(address);
  Wire.write(data, length);
  Wire.endTransmission();
}

// i2c callback function for reads
void i2c_receive(uint8_t address, uint8_t* data, uint8_t length)
{
  Wire.requestFrom(address, length);
  for (uint8_t i = 0; i < length; i++)
  {
    *data = Wire.read();
    data++;
  }
}

void debug_print(char* str)
{
  Serial.println(str);
}
