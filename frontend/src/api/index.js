import axios from 'axios'

// Dynamic API Base URL — uses VITE_API_URL if set in Vercel / .env, otherwise defaults to relative /api/v1
const getBaseUrl = () => {
  const envUrl = import.meta.env.VITE_API_URL
  if (envUrl && typeof envUrl === 'string' && envUrl.trim()) {
    const cleanUrl = envUrl.trim().replace(/\/+$/, '')
    return cleanUrl.endsWith('/api/v1') ? cleanUrl : `${cleanUrl}/api/v1`
  }
  return '/api/v1'
}

const api = axios.create({ baseURL: getBaseUrl(), timeout: 30000 })

api.interceptors.request.use(cfg => {
  const token = localStorage.getItem('cg_token')
  if (token) cfg.headers.Authorization = `Bearer ${token}`
  return cfg
})

api.interceptors.response.use(
  res => res.data,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem('cg_token')
      localStorage.removeItem('cg_user')
      if (!window.location.pathname.includes('/login')) window.location.href = '/login'
    }
    return Promise.reject(new Error(err.response?.data?.detail || err.message || 'Request failed'))
  }
)

export default api

export const authApi = {
  register: d => api.post('/auth/register', d),
  login:    d => api.post('/auth/login', d),
  me:       () => api.get('/auth/me'),
}
export const usersApi = {
  getProfile:    ()  => api.get('/users/me'),
  updateProfile: (d) => api.put('/users/me', d),
  getStats:      ()  => api.get('/users/me/stats'),
}
export const scansApi = {
  create: (file, crop, field='', language='en') => {
    const f = new FormData()
    f.append('image', file); f.append('crop', crop); f.append('field', field); f.append('language', language)
    return api.post('/scans/', f, { headers:{'Content-Type':'multipart/form-data'}, timeout:180000 })
  },
  list:         p  => api.get('/scans/', { params: p }),
  get:          id => api.get(`/scans/${id}`),
  updateStatus: (id,s) => api.patch(`/scans/${id}/status`, { status:s }),
  delete:       id => api.delete(`/scans/${id}`),
  imageUrl:     f  => {
    if (!f) return ''
    if (f.startsWith('http://') || f.startsWith('https://')) return f
    const envUrl = import.meta.env.VITE_API_URL
    const base = (envUrl && typeof envUrl === 'string' && envUrl.trim()) ? envUrl.trim().replace(/\/+$/, '') : ''
    return `${base}/uploads/${f}`
  },
  research: (file, crop='Unknown', language='en') => {
    const f = new FormData()
    f.append('image', file); f.append('crop', crop); f.append('language', language)
    return api.post('/scans/research', f, { headers:{'Content-Type':'multipart/form-data'}, timeout:180000 })
  }
}
export const diagnoseApi = {
  start: (file, crop, field='', language='en') => {
    const f = new FormData()
    f.append('image', file); f.append('crop', crop); f.append('field', field); f.append('language', language)
    return api.post('/diagnose/', f, { headers:{'Content-Type':'multipart/form-data'}, timeout:180000 })
  },
  status: id => api.get(`/diagnose/status/${id}`, { timeout: 30000 })
}
export const treatmentsApi = {
  list:       p    => api.get('/treatments/', { params:p }),
  get:        slug => api.get(`/treatments/${slug}`),
  byDisease:  n    => api.get(`/treatments/disease/${encodeURIComponent(n)}`),
}
export const weatherApi = {
  current:  () => api.get('/weather/current'),
  forecast: () => api.get('/weather/forecast'),
  advisory: () => api.get('/weather/advisory'),
}
export const analyticsApi = {
  dashboard: () => api.get('/analytics/dashboard'),
}
export const notificationsApi = {
  list:        ()  => api.get('/notifications/'),
  markRead:    id  => api.patch(`/notifications/${id}/read`),
  markAllRead: ()  => api.patch('/notifications/read-all'),
  getPrefs:    ()  => api.get('/notifications/preferences'),
  savePrefs:   (d) => api.put('/notifications/preferences', d),
}
export const adminApi = {
  getStats:         ()           => api.get('/admin/stats'),
  getAnalytics:     ()           => api.get('/admin/analytics'),
  getUsers:         (p={})       => api.get('/admin/users', { params: p }),
  getUser:          (id)         => api.get(`/admin/users/${id}`),
  changeRole:       (id, role)   => api.patch(`/admin/users/${id}/role`, { role }),
  disableUser:      (id)         => api.delete(`/admin/users/${id}`),
  getAllScans:       (p={})       => api.get('/admin/scans', { params: p }),
  getModelStatus:   ()           => api.get('/admin/model/status'),
  triggerTraining:   ()          => api.post('/admin/model/trigger-training'),
}
export const researchFeedbackApi = {
  submit:       (body) => api.post('/research-feedback/', body),
  getAnalytics: ()     => api.get('/research-feedback/analytics'),
}
