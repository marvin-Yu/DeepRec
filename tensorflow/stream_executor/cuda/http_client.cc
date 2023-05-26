#include <string>
#include <iostream>
#include <fcntl.h>
#include <errno.h>
#include <sys/uio.h>  // readv
#include <stdint.h>
#include <endian.h>
#include <unistd.h>
#include <map>
#include <fstream>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/env_var.h"
#include "http_client.h"

const std::map<std::string, int>::value_type init_value[] =
{
	std::map<std::string, int>::value_type( "GET", HttpRequest::GET),

	std::map<std::string, int>::value_type( "POST", HttpRequest::POST)
};

const static std::map<std::string, int> kRequestMethodMap(init_value, init_value + (sizeof init_value / sizeof init_value[0]));

static inline uint16_t hostToNetwork16(uint16_t host16)
{
  return htobe16(host16);
}


int sockets::createSocket(sa_family_t family){
  // Call "socket()" to create a (family) socket of the specified type.
  // But also set it to have the 'close on exec' property (if we can)

	int sock;

	//CLOEXEC，即当调用exec（）函数成功后，文件描述符会自动关闭。
	//在以往的内核版本（2.6.23以前）中，需要调用 fcntl(fd, F_SETFD, FD_CLOEXEC) 来设置这个属性。
	//而新版本（2.6.23开始）中，可以在调用open函数的时候，通过 flags 参数设置 CLOEXEC 功能，
#ifdef SOCK_CLOEXEC
  sock = socket(family, SOCK_STREAM|SOCK_CLOEXEC, 0);
  if (sock != -1 || errno != EINVAL) return sock;
  // An "errno" of EINVAL likely means that the system wasn't happy with the SOCK_CLOEXEC; fall through and try again without it:
#endif

  sock = socket(family, SOCK_STREAM, 0);

#ifdef FD_CLOEXEC
  if (sock != -1) fcntl(sock, F_SETFD, FD_CLOEXEC);
#endif
  return sock;
}

int sockets::connect(int sockfd, const struct sockaddr* addr)
{
  return ::connect(sockfd, addr, sizeof(struct sockaddr));
}

void sockets::fromIpPort(const char* ip, uint16_t port,
                         struct sockaddr_in* addr)
{
  addr->sin_family = AF_INET;
  addr->sin_port = hostToNetwork16(port);
  if (::inet_pton(AF_INET, ip, &addr->sin_addr) <= 0)
  {
    VLOG(1) << "sockets::fromIpPort";
  }
}

ssize_t sockets::read(int sockfd, void *buf, size_t count)
{
  return ::read(sockfd, buf, count);
}

ssize_t sockets::readv(int sockfd, const struct iovec *iov, int iovcnt)
{
  return ::readv(sockfd, iov, iovcnt);
}

ssize_t sockets::write(int sockfd, const void *buf, size_t count)
{
  return ::write(sockfd, buf, count);
}

void sockets::close(int sockfd)
{
  if (::close(sockfd) < 0)
  {
    VLOG(1) << "sockets::close";
  }
}

void sockets::delaySecond(int sec)
{
  struct  timeval tv;
  tv.tv_sec = sec;
  tv.tv_usec = 0;
  select(0, NULL, NULL, NULL, &tv);
}



InetAddress::InetAddress(std::string ip, uint16_t port)
{
  ::bzero(&m_addr, sizeof m_addr);
  sockets::fromIpPort(ip.c_str(), port, &m_addr);
}

uint32_t InetAddress::ipNetEndian() const
{
  assert(family() == AF_INET);
  return m_addr.sin_addr.s_addr;
}


HttpRequest::HttpRequest(std::string httpUrl)
  :m_httpUrl(httpUrl)
{

}

HttpRequest::~HttpRequest()
{

}

int HttpRequest::connect()
{
  tensorflow::int64 port;
  tensorflow::ReadInt64FromEnvVar("TF_PTXAS_HTTP_PORT", 8881, &port);
  VLOG(1) << "TF_PTXAS_HTTP_PORT=" << port;
  while(true)
  {
    InetAddress serverAddr = InetAddress("127.0.0.1", (int)port);

    m_sockfd = sockets::createSocket(serverAddr.family());
    if(m_sockfd < 0) VLOG(1) << "HttpRequest::connect() : createSocket error";
    int ret = sockets::connect(m_sockfd, serverAddr.getSockAddr());
    VLOG(1) << "sockfd : " << m_sockfd << "sockets::connect ret : " << ret ;
    if (ret != 0) return ret;

    int savedErrno = (ret == 0) ? 0 : errno;

    switch (savedErrno)
    {
      case 0:
      case EINPROGRESS:
      case EINTR:
      case EISCONN:
        VLOG(1) << "HttpRequest::connect() sockfd : " << m_sockfd << " Successful";
        break;
      default :
        VLOG(1) << "Connect Error ";
        sockets::delaySecond(1);
        continue;
    }

    break;
  }

  VLOG(1) << "HttpRequest::Connector() end";
  return 0;
}

void HttpRequest::setRequestMethod(const std::string &method)
{
	switch(kRequestMethodMap.at(method))
	{
		case HttpRequest::GET :
			m_stream << "GET " << "/" << m_httpUrl << " HTTP/1.1\r\n";
			VLOG(1) << m_stream.str().c_str();
			break;
		case HttpRequest::POST :
			m_stream << "POST "  << "/" << m_httpUrl << " HTTP/1.1\r\n";
			VLOG(1) << m_stream.str().c_str();
			break;
		default :
			VLOG(1) << "No such Method : " << method.c_str();
			break;
	}

	m_stream << "Host: " << "localhost" << "\r\n";
}


void HttpRequest::setRequestProperty(const std::string &key, const std::string &value)
{
	m_stream << key << ": " << value << "\r\n";
}

void HttpRequest::setRequestBody(const std::string &content)
{
	m_stream << content;
}

void HttpRequest::handleRead()
{
	assert(!m_haveHandleHead);
	ssize_t nread = 0;
	ssize_t writtenBytes = 0;

	nread = sockets::read(m_sockfd, m_buffer.beginWrite(), kBufferSize);
	if(nread < 0) VLOG(1) << "sockets::read";
	m_buffer.hasWritten(nread);
	VLOG(1) << "sockets::read(): nread: " << nread << " remain: " << m_buffer.writableBytes();
	size_t remain = kBufferSize - nread;
	while(remain > 0)
	{
		size_t n = sockets::read(m_sockfd, m_buffer.beginWrite(), remain);
		if(n < 0) VLOG(1) << "sockets::read";
		m_buffer.hasWritten(n);
		if(0 == n)
		{
			VLOG(1) << "sockets::read finish";
			break;
		}
		remain = remain - n;
	}
	//VLOG(1) << m_buffer.peek();

	//for(int i = 0; i < nread; i++) printf("%02x%c",(unsigned char)buffer[i],i==nread - 1 ?'\n':' ');
	//VLOG(1) << "handleRead Recv Response : \n" << m_buffer.peek();
	int headsize = 0;
	std::string line;
	std::stringstream ss(m_buffer.peek());
	std::vector<std::string> v;
	getline(ss, line);
	//VLOG(1) << line;
	headsize += line.size() + 1;
	SplitString(line, v, " ");
	//for(int i = 0; i < v.size(); i++) VLOG(1) << v[i] << std::endl;
	if (v.size() < 2) {
          LOG(ERROR) << "socket got message error, v.size=" << v.size();
          m_code = -1;
          return;
        }
	m_code = std::stoi(v[1]);
	if(v[1] != "200")
	{
	  VLOG(1) << "Error Http Server Response : " << v[1].c_str();
	}

	do{
		getline(ss, line);
		headsize += line.size() + 1;  // + 1('\n')
		if(!line.empty()) line.erase(line.end()-1); // remove '/r'
		//VLOG(1) << line;
		v.clear();
		SplitString(line, v, ":");
		if(v.size() == 2){
			m_ackProperty[v[0]] = v[1].erase(0,v[1].find_first_not_of(" "));
		}
	}while(!line.empty());

	VLOG(1) << "Http Head Size is " << headsize;
	std::string res(m_buffer.peek(), headsize);
	VLOG(1) << "Http Response :\n" << res;
	m_buffer.retrieve(headsize);

	m_haveHandleHead = true;

}

void HttpRequest::uploadFile(const std::string& file, const std::string& contentEnd)
{

	FILE* fp = fopen(file.c_str(), "rb");
	if(fp == NULL)
	{
		VLOG(1) << "fopen() File :" << file.c_str() << " Errno";
	}

	bool isEnd = false;
	ssize_t writtenBytes = 0;

	assert(m_buffer.writableBytes() == Buffer::kInitialSize);

	while(!isEnd)
	{
		ssize_t nread = fread(m_buffer.beginWrite(), 1, kBufferSize, fp);
		m_buffer.hasWritten(nread);
		while(m_buffer.writableBytes() > 0)
		{
			VLOG(1) << "file read(): nread: " << nread << " remain: " << m_buffer.writableBytes();
			size_t n = fread(m_buffer.beginWrite(), 1, m_buffer.writableBytes(), fp);
			m_buffer.hasWritten(n);
			if(0 == n)
			{	int err = ferror(fp);
				if(err)
				{
					fprintf(stderr, "fread failed : %s\n", strerror(err));
				}
				VLOG(1) << "sockets::read finish";
				isEnd = true;
				break;
			}
		}

		ssize_t nwrite = sockets::write(m_sockfd, m_buffer.peek(), m_buffer.readableBytes());
		if(nwrite < 0) VLOG(1) << "sockets::write";
		writtenBytes += nwrite;
		VLOG(1) << "sockets::write nread " << m_buffer.readableBytes() << " nwrite " << nwrite << " writtenBytes " << writtenBytes;
		m_buffer.retrieve(nwrite);
	}

	fclose(fp);

	m_buffer.retrieveAll();

	ssize_t n = sockets::write(m_sockfd, contentEnd.c_str(), contentEnd.size());
	if(n < 0) VLOG(1) << "sockets::write";
}

void HttpRequest::downloadFile(const std::string& file)
{
	assert(m_haveHandleHead);

	bool isEnd = false;
	ssize_t nread = 0;
	ssize_t writtenBytes = 0;

	std::ofstream output(file, std::ios::binary);
	if (!output.is_open()){ // 检查文件是否成功打开
		VLOG(1) << "open file error" << file;
	}

	output.write(m_buffer.peek(), m_buffer.readableBytes());
	writtenBytes += m_buffer.readableBytes();
	m_buffer.retrieve(m_buffer.readableBytes());

	VLOG(1) << "Content-Length : " << getResponseProperty("Content-Length");

	while(!isEnd)
	{
		nread = sockets::read(m_sockfd, m_buffer.beginWrite(), kBufferSize);
		if(nread < 0) VLOG(1) << "sockets::read";
		m_buffer.hasWritten(nread);
		VLOG(1) << "sockets::read(): nread: " << nread << " remain: " << m_buffer.writableBytes() << " writtenBytes: " << writtenBytes;
		size_t remain = kBufferSize - nread;
		while(remain > 0)
		{
			size_t n = sockets::read(m_sockfd, m_buffer.beginWrite(), remain);
			if(n < 0) VLOG(1) << "sockets::read";
			m_buffer.hasWritten(n);
			if(0 == n)
			{
				VLOG(1) << "sockets::read finish";
				isEnd = true;
				break;
			}
			remain = remain - n;
		}

		output.write(m_buffer.peek(), m_buffer.readableBytes());
		writtenBytes += m_buffer.readableBytes();
		m_buffer.retrieve(m_buffer.readableBytes());
	}
	VLOG(1) << " writtenBytes " << writtenBytes;

	output.close();
	sockets::close(m_sockfd);
}

void HttpRequest::SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));

    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
}


