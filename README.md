
## Hvordan kjøre scriptet

### Kjør alle enhetstester (11/11 OK):
python C:/dev/jpeg.py --test

### Generer alle 10 plott for et bilde:
python C:/dev/jpeg.py --plot C:/dev/bilde.jpg --quality 50 --outdir C:/dev/jpeg_plots

### Kjør full demonstrasjon (både tester og plott):
python C:/dev/jpeg.py --demo


<img width="3600" height="1900" alt="00_jpeg_full_pipeline_overview" src="https://github.com/user-attachments/assets/e858bb5e-755e-48be-b208-226575937139" />
<img width="3000" height="1500" alt="01_color_space_conversion" src="https://github.com/user-attachments/assets/b13ed210-d27f-4f16-955d-f66225e4b0d8" />
<img width="2600" height="1200" alt="02_block_partition_8x8" src="https://github.com/user-attachments/assets/a2423903-0155-4cbd-8552-8e64b5b0be18" />
<img width="3000" height="1000" alt="03_level_shifting" src="https://github.com/user-attachments/assets/0b854a57-e044-4741-a333-c3a776df34b2" />
<img width="1900" height="1900" alt="04_dct_basis_functions" src="https://github.com/user-attachments/assets/f2168028-9b50-41f4-93a5-73fc3a844cfd" />
<img width="3200" height="1040" alt="05_dct_coefficients" src="https://github.com/user-attachments/assets/222d764d-e2c4-4900-ba84-f17ce5e603c7" />
<img width="3600" height="960" alt="06_quantization_and_sparsity" src="https://github.com/user-attachments/assets/8a93e6fe-1807-43a0-85eb-209e555c789e" />
<img width="3000" height="1300" alt="07_zigzag_scanning_and_rle" src="https://github.com/user-attachments/assets/d1aae162-15b6-43e0-b167-7cd6a116af77" />
<img width="3400" height="960" alt="08_inverse_dct_and_error" src="https://github.com/user-attachments/assets/7374c5b4-2b8e-4da9-9dab-1f13e7989823" />
<img width="3200" height="1500" alt="09_quality_comparison_and_rate_distortion" src="https://github.com/user-attachments/assets/aad4eae7-ae34-4ee2-a50e-41136c165f4d" />

