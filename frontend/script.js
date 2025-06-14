async function verify() {
  const img1 = document.getElementById('img1').files[0];
  const img2 = document.getElementById('img2').files[0];

  const formData = new FormData();
  formData.append("img1", img1);
  formData.append("img2", img2);

  const response = await fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  document.getElementById('result').innerText = 
    `Similarity: ${data.similarity_score} → ${data.verdict}`;
}
