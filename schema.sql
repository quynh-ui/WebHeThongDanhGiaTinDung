-- Cập nhật bảng users nếu thiếu cột
ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active'; -- 'active', 'locked', 'pending'
ALTER TABLE users ADD COLUMN status_reason TEXT;
ALTER TABLE users ADD COLUMN last_login TIMESTAMP;

-- Bảng ghi nhận lịch sử hoạt động
CREATE TABLE IF NOT EXISTS user_activity_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action TEXT NOT NULL,
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);