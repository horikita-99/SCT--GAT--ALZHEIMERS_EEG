# ====================== Graph construction ======================
def row_to_graph(row):
    x = []
    for ch in channels:
        x.append([
            row[f"{ch}_delta"], row[f"{ch}_theta"],
            row[f"{ch}_alpha"], row[f"{ch}_beta"],
            row[f"{ch}_gamma"]
        ])
    x = torch.tensor(x, dtype=torch.float)
    edge_index = get_knn_edge_index(x.numpy(), k=4)  # 🔄 KNN-based edge construction
    #edge_index = get_dynamic_edge_index(x.numpy(), threshold=0.5)
    y = torch.tensor([label_map[row['label']]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

graphs = [row_to_graph(row) for _, row in df.iterrows()]

# ====================== Split and Loaders ======================
train_graphs, test_graphs = train_test_split(
    graphs, test_size=0.2, stratify=df['label'], random_state=42
)
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=16)
# ====================== 6. GAT Model ======================
class EEG_GAT_Dropout(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=25, heads=4, dropout=0.1, num_classes=3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, dropout=dropout)
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        x = global_max_pool(x, batch)
        return self.fc(x)

model = EEG_GAT_Dropout().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

# ====================== 7. Training ======================
for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"📊 Epoch {epoch}, Loss: {total_loss:.4f}")

========================Evaluation -- AD/CN  ======================
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"\n✅ GAT Test Accuracy: {acc:.2%}")
print(classification_report(y_true, y_pred, target_names=['AD', 'CN']))
