// Interactive Client-side Script for Customer Churn Prediction

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("churnForm");
  const tenureInput = document.getElementById("tenure");
  const tenureVal = document.getElementById("tenureVal");
  const monthlyInput = document.getElementById("MonthlyCharges");
  const monthlyVal = document.getElementById("monthlyVal");
  const totalInput = document.getElementById("TotalCharges");
  const predictBtn = document.getElementById("predictBtn");
  const dialProgress = document.getElementById("dialProgress");
  const dialPercent = document.getElementById("dialPercent");
  const riskBadge = document.getElementById("riskBadge");
  const predictionLabel = document.getElementById("predictionLabel");
  const dbStatus = document.getElementById("dbStatus");
  const factorsContainer = document.getElementById("factorsContainer");
  const actionsContainer = document.getElementById("actionsContainer");

  // Sync Slider values
  if (tenureInput && tenureVal) {
    tenureInput.addEventListener("input", (e) => {
      tenureVal.textContent = e.target.value;
      autoCalcTotal();
    });
  }

  if (monthlyInput && monthlyVal) {
    monthlyInput.addEventListener("input", (e) => {
      monthlyVal.textContent = parseFloat(e.target.value).toFixed(2);
      autoCalcTotal();
    });
  }

  function autoCalcTotal() {
    const tenure = parseFloat(tenureInput.value) || 0;
    const monthly = parseFloat(monthlyInput.value) || 0;
    if (totalInput) {
      totalInput.value = (tenure * monthly).toFixed(2);
    }
  }

  // Presets definition
  const presets = {
    highRisk: {
      gender: "Male",
      SeniorCitizen: "0",
      Partner: "No",
      Dependents: "No",
      tenure: "2",
      PhoneService: "Yes",
      MultipleLines: "No",
      InternetService: "Fiber optic",
      OnlineSecurity: "No",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "Yes",
      StreamingMovies: "Yes",
      Contract: "Month-to-month",
      PaperlessBilling: "Yes",
      PaymentMethod: "Electronic check",
      MonthlyCharges: "89.50",
      TotalCharges: "179.00",
    },
    loyalEnterprise: {
      gender: "Female",
      SeniorCitizen: "0",
      Partner: "Yes",
      Dependents: "Yes",
      tenure: "64",
      PhoneService: "Yes",
      MultipleLines: "Yes",
      InternetService: "DSL",
      OnlineSecurity: "Yes",
      OnlineBackup: "Yes",
      DeviceProtection: "Yes",
      TechSupport: "Yes",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: "Two year",
      PaperlessBilling: "No",
      PaymentMethod: "Credit card (automatic)",
      MonthlyCharges: "54.20",
      TotalCharges: "3468.80",
    },
    fiberAtRisk: {
      gender: "Male",
      SeniorCitizen: "1",
      Partner: "No",
      Dependents: "No",
      tenure: "8",
      PhoneService: "Yes",
      MultipleLines: "Yes",
      InternetService: "Fiber optic",
      OnlineSecurity: "No",
      OnlineBackup: "Yes",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "Yes",
      StreamingMovies: "Yes",
      Contract: "Month-to-month",
      PaperlessBilling: "Yes",
      PaymentMethod: "Electronic check",
      MonthlyCharges: "98.75",
      TotalCharges: "790.00",
    },
    budgetBasic: {
      gender: "Female",
      SeniorCitizen: "0",
      Partner: "No",
      Dependents: "No",
      tenure: "28",
      PhoneService: "Yes",
      MultipleLines: "No",
      InternetService: "DSL",
      OnlineSecurity: "Yes",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "Yes",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: "One year",
      PaperlessBilling: "Yes",
      PaymentMethod: "Bank transfer (automatic)",
      MonthlyCharges: "35.50",
      TotalCharges: "994.00",
    },
  };

  window.loadPreset = function (presetKey) {
    const data = presets[presetKey];
    if (!data) return;

    for (const [key, val] of Object.entries(data)) {
      const elem = document.getElementById(key);
      if (elem) {
        elem.value = val;
      }
    }

    if (tenureVal) tenureVal.textContent = data.tenure;
    if (monthlyVal) monthlyVal.textContent = parseFloat(data.MonthlyCharges).toFixed(2);

    // Trigger prediction automatically
    runPrediction();
  };

  // Form submission / prediction trigger
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      runPrediction();
    });
  }

  async function runPrediction() {
    if (!predictBtn) return;
    predictBtn.disabled = true;
    predictBtn.innerHTML = `<span>Computing Probability...</span>`;

    const formData = new FormData(form);
    const payload = {};

    formData.forEach((value, key) => {
      payload[key] = value;
    });

    // Parse numeric fields
    payload.tenure = parseInt(payload.tenure, 10);
    payload.MonthlyCharges = parseFloat(payload.MonthlyCharges);
    payload.TotalCharges = parseFloat(payload.TotalCharges) || payload.tenure * payload.MonthlyCharges;
    payload.SeniorCitizen = parseInt(payload.SeniorCitizen, 10);

    // Handle Internet Service cascading for add-on options
    if (payload.InternetService === "No") {
      ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"].forEach(
        (addon) => {
          payload[addon] = "No internet service";
        }
      );
    }

    // Handle Phone Service cascading
    if (payload.PhoneService === "No") {
      payload.MultipleLines = "No phone service";
    }

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok || !data.success) {
        alert("Prediction error: " + (data.error || "Unknown server response"));
        return;
      }

      renderPredictionResult(data);
    } catch (err) {
      console.error(err);
      alert("Failed to communicate with prediction API: " + err.message);
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerHTML = `<span>Run Prediction</span>`;
    }
  }

  function renderPredictionResult(data) {
    const prob = data.churn_probability;
    const pct = (prob * 100).toFixed(1);

    // Update Dial text
    if (dialPercent) {
      dialPercent.textContent = `${pct}%`;
    }

    // Update Dial progress animation
    // Circle circumference = 2 * PI * 70 approx 440
    const circumference = 440;
    const offset = circumference - (circumference * prob);
    if (dialProgress) {
      dialProgress.style.strokeDashoffset = offset;

      // Color coding
      if (prob < 0.35) {
        dialProgress.style.stroke = "var(--risk-low)";
      } else if (prob < 0.65) {
        dialProgress.style.stroke = "var(--risk-medium)";
      } else {
        dialProgress.style.stroke = "var(--risk-high)";
      }
    }

    // Update Risk Badge
    if (riskBadge) {
      riskBadge.className = `risk-badge ${data.risk_level.toLowerCase()}`;
      riskBadge.textContent = `${data.risk_level} Churn Risk`;
    }

    // Update Prediction Label
    if (predictionLabel) {
      predictionLabel.textContent = data.prediction;
      predictionLabel.style.color = data.prediction === "Likely to churn" ? "#f87171" : "#34d399";
    }

    // Update Database Status
    if (dbStatus) {
      if (data.saved) {
        dbStatus.innerHTML = `<span style="color: #34d399;">✓ Logged to Database</span>`;
      } else {
        dbStatus.innerHTML = `<span style="color: #fbbf24;">⚠ Database offline: ${data.save_error || "unreachable"}</span>`;
      }
    // Update Financial Revenue Impact
    const revElem = document.getElementById("revenueAtRisk");
    const saveElem = document.getElementById("retentionSavings");
    if (revElem && data.financial_impact) {
      revElem.textContent = `$${parseFloat(data.financial_impact.annual_revenue_at_risk || 0).toFixed(2)}`;
    }
    if (saveElem && data.financial_impact) {
      saveElem.textContent = `$${parseFloat(data.financial_impact.simulated_retention_savings || 0).toFixed(2)}`;
    }

    // Render Factor Drivers
    if (factorsContainer && data.risk_factors) {
      factorsContainer.innerHTML = "";
      if (data.risk_factors.length === 0) {
        factorsContainer.innerHTML = `<p style="font-size: 0.8rem; color: var(--text-muted);">Neutral profile drivers.</p>`;
      } else {
        data.risk_factors.forEach((f) => {
          const div = document.createElement("div");
          div.className = "factor-card";
          const meterWidth = f.impact.includes("High") ? "85%" : f.impact.includes("Moderate") ? "60%" : "40%";
          div.innerHTML = `
            <div style="flex: 1;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                <strong style="color: #fff; font-size: 0.82rem;">${f.feature}</strong>
                <span class="factor-badge ${f.type}">${f.impact}</span>
              </div>
              <span style="color: var(--text-secondary); font-size: 0.78rem;">${f.description}</span>
              <div class="factor-meter-bg">
                <div class="factor-meter-fill ${f.type}" style="width: ${meterWidth};"></div>
              </div>
            </div>
          `;
          factorsContainer.appendChild(div);
        });
      }
    }

    // Render Retention Playbook Actions
    if (actionsContainer && data.retention_actions) {
      actionsContainer.innerHTML = "";
      data.retention_actions.forEach((act) => {
        const div = document.createElement("div");
        div.className = "action-card";
        div.innerHTML = `
          <span style="color: var(--accent); font-weight: bold;">➜</span>
          <span style="color: var(--text-primary); font-size: 0.82rem;">${act}</span>
        `;
        actionsContainer.appendChild(div);
      });
    }
  }

  // Batch Form Handler
  const batchForm = document.getElementById("batchForm");
  const batchBtn = document.getElementById("batchBtn");
  const batchResults = document.getElementById("batchResults");
  const downloadBatchCsvBtn = document.getElementById("downloadBatchCsvBtn");

  if (batchForm) {
    batchForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById("batchFile");
      if (!fileInput.files.length) {
        alert("Please select a CSV file first.");
        return;
      }

      batchBtn.disabled = true;
      batchBtn.innerHTML = `<span>Processing Batch Scoring...</span>`;

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      try {
        const res = await fetch("/api/batch-predict?format=json", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        if (!res.ok || !data.success) {
          alert("Batch prediction failed: " + (data.error || "Unknown error"));
          return;
        }

        const summary = data.summary;
        document.getElementById("batchTotal").textContent = summary.total;
        document.getElementById("batchHigh").textContent = summary.high_risk;
        document.getElementById("batchChurners").textContent = summary.predicted_churners;
        batchResults.style.display = "block";
      } catch (err) {
        alert("Batch error: " + err.message);
      } finally {
        batchBtn.disabled = false;
        batchBtn.innerHTML = `<span>Upload & Analyze Batch</span>`;
      }
    });
  }

  if (downloadBatchCsvBtn) {
    downloadBatchCsvBtn.addEventListener("click", () => {
      const fileInput = document.getElementById("batchFile");
      if (!fileInput.files.length) return;

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      fetch("/api/batch-predict", {
        method: "POST",
        body: formData,
      })
        .then((res) => res.blob())
        .then((blob) => {
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "batch_churn_predictions.csv";
          document.body.appendChild(a);
          a.click();
          a.remove();
        })
        .catch((err) => alert("Download failed: " + err.message));
    });
  }

  // Trigger initial prediction with default preset on load
  if (form) {
    runPrediction();
  }
});
